from typing import Optional
from functools import partial

from absl import logging

import flax.training.train_state
import flax.training.checkpoints
import flax.linen as nn
import jax.numpy as jnp
import jax.random
import optax
import rlax

import nle.nethack


from ..utils.profiling import timeit
from .trainable_agent import TrainableAgentBase
from ..neural import TransformerNet, CrossTransformerNet, DenseNet


class NethackPerceiverModel(nn.Module):
    num_actions: int
    glyph_embedding_dim: int = 64
    num_memory_units: int = 128
    memory_dim: int = 64
    num_perceiver_blocks: int = 2
    num_perceiver_self_attention_subblocks: int = 2
    transformer_dropout: float = 0.1
    memory_update_num_heads: int = 8
    map_attention_num_heads: int = 2
    transformer_fc_inner_dim: int = 256
    num_policy_network_blocks: int = 4
    num_value_network_blocks: int = 4
    deterministic: Optional[bool] = None

    def setup(self):
        self._glyph_pos_embedding = nn.Embed(
            num_embeddings=nle.nethack.DUNGEON_SHAPE[0] * nle.nethack.DUNGEON_SHAPE[1],
            features=self.glyph_embedding_dim
        )
        self._glyph_embedding = nn.Embed(
            num_embeddings=nle.nethack.MAX_GLYPH + 1,
            features=self.glyph_embedding_dim
        )
        self._memory_embedding = nn.Embed(
            num_embeddings=self.num_memory_units,
            features=self.memory_dim
        )
        self._memory_update_blocks = [
            TransformerNet(
                num_blocks=self.num_perceiver_self_attention_subblocks,
                dim=self.memory_dim,
                fc_inner_dim=self.transformer_fc_inner_dim,
                num_heads=self.memory_update_num_heads,
                dropout_rate=self.transformer_dropout,
                deterministic=self.deterministic,
                name='{}/perceiver_self_attention_block_{}'.format(self.name, block_idx)
            )
            for block_idx in range(self.num_perceiver_blocks)
        ]
        self._map_attention_blocks = [
            CrossTransformerNet(
                num_blocks=1,
                dim=self.memory_dim,
                fc_inner_dim=self.transformer_fc_inner_dim,
                num_heads=self.map_attention_num_heads,
                dropout_rate=self.transformer_dropout,
                deterministic=self.deterministic,
                name='{}/perceiver_map_attention_block_{}'.format(self.name, block_idx),
            )
            for block_idx in range(self.num_perceiver_blocks)
        ]
        self._policy_network = DenseNet(
            num_blocks=self.num_policy_network_blocks, dim=self.memory_dim, output_dim=self.num_actions,
        )
        self._value_network = DenseNet(
            num_blocks=self.num_value_network_blocks, dim=self.memory_dim, output_dim=1,
        )

    def __call__(self, observations_batch, deterministic=None, rng=None):
        deterministic = nn.module.merge_param('deterministic', self.deterministic, deterministic)
        if rng is None:
            rng = self.make_rng('nethack_transformer_model')

        glyphs = observations_batch['glyphs']
        batch_size = glyphs.shape[0]

        memory_indices = jnp.arange(0, self.num_memory_units, dtype=jnp.int32)
        memory = self._memory_embedding(memory_indices)  # TODO: add memory recurrence
        memory = jnp.tile(memory, reps=[batch_size, 1, 1])

        glyphs = jnp.reshape(glyphs, newshape=(glyphs.shape[0], -1))
        glyphs_embeddings = self._glyph_embedding(glyphs)

        glyph_pos_indices = jnp.arange(
            0, nle.nethack.DUNGEON_SHAPE[0] * nle.nethack.DUNGEON_SHAPE[1], dtype=jnp.int32)
        glyph_pos_embeddings = self._glyph_pos_embedding(glyph_pos_indices)
        glyphs_embeddings += glyph_pos_embeddings

        for block_idx in range(self.num_perceiver_blocks):
            rng, subkey = jax.random.split(rng)
            memory = self._map_attention_blocks[block_idx](
                memory, glyphs_embeddings, deterministic=deterministic, rng=subkey)

            rng, subkey = jax.random.split(rng)
            memory = self._memory_update_blocks[block_idx](
                memory, deterministic=deterministic, rng=subkey)

        # TODO: use pooling instead of fixed memory cell?

        action_logits = self._policy_network(memory[:, 0, :])
        action_logprobs = jax.nn.log_softmax(action_logits, axis=-1)

        state_value = self._value_network(memory[:, 1, :])
        state_value = jnp.squeeze(state_value, axis=-1)

        return action_logprobs, state_value


class NethackTransformerAgent(TrainableAgentBase):
    CONFIG = flax.core.frozen_dict.FrozenDict({
        'model_batch_size': 32,
        'value_function_loss_weight': 1.0,
        'lr': 1e-3,
    })

    class TrainState(flax.training.train_state.TrainState):
        step_index: int = None

    def __init__(self, *args, config=None, **kwargs):
        super(NethackTransformerAgent, self).__init__(*args, **kwargs)

        self._random_key = jax.random.PRNGKey(0)
        self._config = self.CONFIG.copy(config or {})
        self._model, self._train_state = self._build_model()

    # TODO: can this function be made jit-able?
    def _build_model_batch(self, trajectory_batch):
        model_batch_size = self._config['model_batch_size']

        observations_unstacked = {
            'glyphs': []
        }
        targets_unstacked = {
            'actions': [],
            'rewards': [],
            'dest_state_values': [],
        }
        target_dtypes = {
            'actions': jnp.int32,
            'rewards': jnp.float32,
            'dest_state_values': jnp.float32,
        }

        # We need this to sample short trajectories less often
        def num_available_src_states(trajectory):
            if trajectory.transitions[-1].done:
                return len(trajectory.transitions)
            else:
                # If the trajectory doesn't end in a terminal state,
                # we won't have a value estimate for the last state so we can't use the transition to it.
                return len(trajectory.transitions) - 1

        trajectory_freqs = jnp.array(
            [num_available_src_states(t) for t in trajectory_batch.trajectories],
            dtype=jnp.int32)
        trajectory_logprobs = jnp.log(trajectory_freqs / jnp.sum(trajectory_freqs, dtype=jnp.float32))

        for example_index in range(model_batch_size):
            trajectory_index = jax.random.categorical(self._next_random_key(), trajectory_logprobs)
            trajectory = trajectory_batch.trajectories[trajectory_index]

            transition_index = jax.random.randint(
                self._next_random_key(), shape=(), minval=0, maxval=trajectory_freqs[trajectory_index])

            # The src state of the transition is either an initial state or a dest state of the previous transition
            src_state = (
                trajectory.initial_state if transition_index == 0
                else trajectory.transitions[transition_index - 1].observation
            )
            observations_unstacked['glyphs'].append(src_state['glyphs'])

            targets_unstacked['actions'].append(trajectory.transitions[transition_index].action)
            targets_unstacked['rewards'].append(trajectory.transitions[transition_index].reward)

            if transition_index < len(trajectory.transitions) - 1:
                # We have a value computed for the next state
                dest_state_value = trajectory.transitions[transition_index + 1].metadata['state_values']
            else:
                assert trajectory.transitions[transition_index].done  # We can only do it for the terminal state
                dest_state_value = 0.0  # We can't get any reward from a terminal state
            targets_unstacked['dest_state_values'].append(dest_state_value)

        observations = {
            key: jnp.stack(values, axis=0)
            for key, values in observations_unstacked.items()
        }
        targets = {
            key: jnp.array(values, dtype=target_dtypes[key])
            for key, values in targets_unstacked.items()
        }

        return observations, targets

    def _next_random_key(self):
        self._random_key, subkey = jax.random.split(self._random_key)
        return subkey

    def _build_model(self):
        observations_batch = {
            'glyphs': jnp.zeros(
                shape=(1, nle.nethack.DUNGEON_SHAPE[0], nle.nethack.DUNGEON_SHAPE[1]),
                dtype=jnp.int32
            )
        }
        return self._build_model_for_batch(observations_batch)

    def _build_model_for_batch(self, observations_batch):
        model = NethackPerceiverModel(num_actions=self.action_space.n)
        model_params = model.init(
            self._next_random_key(), observations_batch, deterministic=False, rng=self._next_random_key())

        optimizer = optax.adam(learning_rate=self._config['lr'])

        train_state = self.TrainState.create(apply_fn=model.apply, params=model_params, tx=optimizer)

        return model, train_state

    def try_load_from_checkpoint(self, path):
        """
        Loads a checkpoint from the given path if there are any.

        :param path: The path to checkpoints.
        :return: The index of the step that should be next.
        """
        checkpoint_path = flax.training.checkpoints.latest_checkpoint(path)
        if checkpoint_path is None:
            logging.warning('No checkpoints available at {}'.format(path))
            return 0
        else:
            logging.info('State will be loaded from checkpoint {}'.format(checkpoint_path))
            self._train_state = flax.training.checkpoints.restore_checkpoint(checkpoint_path, self._train_state)
            return self._train_state.step_index + 1

    def save_to_checkpoint(self, path, step):
        self._train_state = self._train_state.replace(step_index=step)
        flax.training.checkpoints.save_checkpoint(
            path, self._train_state, step=step, keep=1, overwrite=True)

    @partial(jax.jit, static_argnums=(0,))
    def _forward_step(self, train_state, observation_batch, rng):
        rng1, rng2 = jax.random.split(rng)
        log_action_probs, state_values = train_state.apply_fn(
            train_state.params, observation_batch, deterministic=True, rng=rng1)
        metadata = {
            'log_action_probs': log_action_probs,
            'state_values': state_values,
        }
        # TODO: we need some form of exploration or policy entropy regularization
        selected_actions = jax.random.categorical(rng2, log_action_probs)
        return selected_actions, metadata

    @partial(jax.jit, static_argnums=(0,))
    def _train_step(self, train_state, observations_batch, targets_batch, rng):
        def loss_function(params, rng):
            # TODO(hr0nix): it should come from trajectory in targets_batch, but how would you do backprop?
            log_action_probs, state_values = train_state.apply_fn(
                params, observations_batch, deterministic=False, rng=rng)
            ones = jnp.ones(shape=state_values.shape)
            batch_td_learning = jax.vmap(rlax.td_learning, in_axes=0)
            td_error = batch_td_learning(
                state_values, targets_batch['rewards'], ones, targets_batch['dest_state_values'])
            policy_gradient_loss = rlax.policy_gradient_loss(
                log_action_probs, targets_batch['actions'], td_error, ones)
            value_function_loss = self._config['value_function_loss_weight'] * jnp.mean(td_error ** 2)
            return policy_gradient_loss + value_function_loss

        loss_function_grad = jax.grad(loss_function)
        grads = loss_function_grad(train_state.params, rng=rng)
        return train_state.apply_gradients(grads=grads)

    def act(self, observation_batch):
        return self._forward_step(self._train_state, observation_batch, self._next_random_key())

    def train_on_batch(self, trajectory_batch):
        # TODO: replace pair sampling by recurrence
        observations_batch, targets_batch = self._build_model_batch(trajectory_batch)
        self._train_state = self._train_step(
            self._train_state, observations_batch, targets_batch, self._next_random_key())
