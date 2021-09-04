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
from rlax._src import distributions

import nle.nethack


from ..utils.profiling import timeit
from ..utils.pytree import dict_update
from .trainable_agent import TrainableAgentBase
from ..neural import TransformerNet, CrossTransformerNet, DenseNet
from ..neural.optimization import clip_gradient_by_norm


class NethackPerceiverModel(nn.Module):
    num_actions: int
    glyph_embedding_dim: int = 64
    num_memory_units: int = 128
    memory_dim: int = 64
    num_bl_stats_blocks: int = 2
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
        self._bl_stats_network = DenseNet(
            num_blocks=self.num_bl_stats_blocks, dim=self.memory_dim, output_dim=self.memory_dim,
        )
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

        bl_stats = observations_batch['blstats']
        bl_stats = self._bl_stats_network(bl_stats)

        memory_indices = jnp.arange(0, self.num_memory_units, dtype=jnp.int32)
        memory = self._memory_embedding(memory_indices)  # TODO: add memory recurrence
        memory = jnp.tile(memory, reps=[batch_size, 1, 1])

        memory = memory + jnp.expand_dims(bl_stats, axis=1)  # Add global features to every memory cell

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
        'value_function_loss_weight': 1.0,
        'entropy_regularizer_weight': 0.0,
        'lr': 1e-3,
        'discount_factor': 0.99,
        'gae_lambda': 0.95,
        'gradient_clipnorm': None,
        'num_minibatches_per_train_step': 100,
        'minibatch_size': 64,
        'ppo_eps': 0.25,
        'perceiver_config': {}
    })

    class TrainState(flax.training.train_state.TrainState):
        step_index: int = None

    def __init__(self, *args, config=None, **kwargs):
        super(NethackTransformerAgent, self).__init__(*args, **kwargs)

        self._random_key = jax.random.PRNGKey(31337)
        self._config = self.CONFIG.copy(config or {})
        self._model, self._train_state = self._build_model()

    def _next_random_key(self):
        self._random_key, subkey = jax.random.split(self._random_key)
        return subkey

    def _build_model(self):
        observations_batch = {
            key: jnp.zeros(
                shape=(1,) + desc.shape,
                dtype=desc.dtype
            )
            for key, desc in self.observation_space.spaces.items()
        }
        return self._build_model_for_batch(observations_batch)

    def _build_model_for_batch(self, observations_batch):
        model = NethackPerceiverModel(num_actions=self.action_space.n, **self._config['perceiver_config'])
        model_params = model.init(
            self._next_random_key(), observations_batch, deterministic=False, rng=self._next_random_key())

        optimizer = optax.rmsprop(learning_rate=self._config['lr'])

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
        selected_actions = jax.random.categorical(rng2, log_action_probs)
        return selected_actions, metadata

    @partial(jax.jit, static_argnums=(0,))
    def _train_on_minibatch(self, train_state, trajectory_minibatch, rng):
        def loss_function(params, train_state, trajectory_minibatch, rng):
            log_action_probs, state_values = train_state.apply_fn(
                params, trajectory_minibatch['observations'], deterministic=False, rng=rng)

            minibatch_range = jnp.arange(0, log_action_probs.shape[0])
            log_sampled_action_probs = log_action_probs[minibatch_range, trajectory_minibatch['actions']]
            log_sampled_prev_action_probs = trajectory_minibatch['metadata']['log_action_probs'][
                minibatch_range, trajectory_minibatch['actions']]
            action_prob_ratio = jnp.exp(log_sampled_action_probs - log_sampled_prev_action_probs)
            actor_loss_1 = -action_prob_ratio * trajectory_minibatch['advantage']
            actor_loss_2 = -jnp.clip(
                action_prob_ratio,
                1.0 - self._config['ppo_eps'],
                1.0 + self._config['ppo_eps'],
            ) * trajectory_minibatch['advantage']
            ppo_loss = jnp.maximum(actor_loss_1, actor_loss_2).mean()

            value_function_loss = self._config['value_function_loss_weight'] * jnp.mean(
                0.5 * (state_values - trajectory_minibatch['value_targets']) ** 2)

            entropy_regularizer_loss = self._config['entropy_regularizer_weight'] * rlax.entropy_loss(
                log_action_probs, jnp.ones_like(trajectory_minibatch['advantage']))

            return ppo_loss + value_function_loss + entropy_regularizer_loss, {
                'ppo_loss': ppo_loss,
                'value_function_loss': value_function_loss,
                'entropy_regularizer_loss': entropy_regularizer_loss,
            }

        grad_and_stats = jax.grad(loss_function, argnums=0, has_aux=True)
        grads, stats = grad_and_stats(train_state.params, train_state, trajectory_minibatch, rng)

        if self._config['gradient_clipnorm'] is not None:
            grads = clip_gradient_by_norm(grads, self._config['gradient_clipnorm'])

        return train_state.apply_gradients(grads=grads), stats

    @partial(jax.jit, static_argnums=(0,))
    def _sample_minibatch(self, trajectory_batch, rng):
        key1, key2 = jax.random.split(rng)

        minibatch_size = self._config['minibatch_size']
        some_tensor = jax.tree_leaves(trajectory_batch)[0]
        num_trajectories = some_tensor.shape[0]
        num_timestamps = some_tensor.shape[1]

        trajectory_indices = jax.random.randint(key1, (minibatch_size,), 0, num_trajectories)
        timestamp_indices = jax.random.randint(key2, (minibatch_size,), 0, num_timestamps)

        # Note that we replace trajectory and timestamp dimensions by minibatch dimension here
        return jax.tree_map(
            lambda leaf: leaf[trajectory_indices, timestamp_indices, ...],
            trajectory_batch,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _compute_advantage_and_value_targets(self, trajectory_batch):
        def per_trajectory_advantage(rewards, discounts, state_values):
            return rlax.truncated_generalized_advantage_estimation(
                rewards[:-1], discounts[:-1], self._config['gae_lambda'], state_values)

        per_batch_advantage = jax.vmap(per_trajectory_advantage, in_axes=0)
        discounts = trajectory_batch['done'] * self._config['discount_factor']
        advantage = per_batch_advantage(
            trajectory_batch['rewards'], discounts, trajectory_batch['metadata']['state_values'])
        value_targets = advantage + trajectory_batch['metadata']['state_values'][:, :-1]
        return advantage, value_targets

    def _train_step(self, train_state, trajectory_batch, rng):
        advantage, value_targets = self._compute_advantage_and_value_targets(trajectory_batch)
        # Get rid of states we don't have GAE estimates for
        trajectory_batch = jax.tree_map(lambda l: l[:, :-1, ...], trajectory_batch)
        trajectory_batch = dict_update(trajectory_batch, {
            'advantage': advantage,
            'value_targets': value_targets,
        })
        train_stats_sum = None
        for _ in range(self._config['num_minibatches_per_train_step']):
            rng, subkey1, subkey2 = jax.random.split(rng, 3)
            trajectory_minibatch = self._sample_minibatch(trajectory_batch, subkey1)
            train_state, train_stats = self._train_on_minibatch(train_state, trajectory_minibatch, subkey2)
            train_stats_sum = train_stats if train_stats_sum is None else jax.tree_map(
                jnp.add, train_stats_sum, train_stats)

        # TODO: jit stats computation?
        stats = {
            'state_value': jnp.mean(trajectory_batch['metadata']['state_values']),
            'advantage': jnp.mean(trajectory_batch['advantage']),
            'value_target': jnp.mean(trajectory_batch['value_targets']),
            'policy_entropy': jnp.mean(
                distributions.softmax().entropy(trajectory_batch['metadata']['log_action_probs']))
        }
        stats = dict_update(stats, {
            k: v / self._config['num_minibatches_per_train_step']
            for k, v in train_stats_sum.items()
        })

        return train_state, stats

    def act(self, observation_batch):
        return self._forward_step(self._train_state, observation_batch, self._next_random_key())

    def train_on_batch(self, trajectory_batch):
        self._train_state, stats = self._train_step(self._train_state, trajectory_batch.buffer, self._next_random_key())
        return stats
