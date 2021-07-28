import copy
from typing import Optional

import flax.linen as nn
import jax.numpy as jnp
import jax.random
import optax
import rlax

from .trainable_agent import TrainableAgentBase
from ..neural import TransformerNet, CrossTransformerNet, DenseNet


class NethackAgent(TrainableAgentBase):
    MAX_GLYPH = 5991
    MAP_ROWS = 21
    MAP_COLS = 79
    FEATURES = 25
    MAX_MESSAGE_LENGTH = 256
    MAX_INVENTORY_LENGTH = 55
    NUM_ACTIONS = 98


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
            num_embeddings=NethackAgent.MAP_ROWS * NethackAgent.MAP_COLS,
            features=self.glyph_embedding_dim
        )
        self._glyph_embedding = nn.Embed(
            num_embeddings=NethackAgent.MAX_GLYPH + 1,
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

        glyph_pos_indices = jnp.arange(0, NethackAgent.MAP_ROWS * NethackAgent.MAP_COLS, dtype=jnp.int32)
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


class NethackTransformerAgent(NethackAgent):
    CONFIG = {
        'model_batch_size': 32,
        'value_function_loss_weight': 1.0,
        'lr': 1e-3,
    }

    def __init__(self, *args, config=None, **kwargs):
        super(NethackTransformerAgent, self).__init__(*args, **kwargs)
        
        self._config = copy.deepcopy(self.CONFIG)
        self._config.update(config or {})

        self._model = None
        self._model_params = None
        self._optimizer = None
        self._optimizer_state = None

        self._random_key = jax.random.PRNGKey(0)

    def _build_model_batch(self, trajectory_batch):
        rewards_to_go = [
            t.compute_rewards_to_go()
            for t in trajectory_batch.trajectories
        ]

        model_batch_size = self._config['model_batch_size']

        observations_unstacked = {
            'glyphs': []
        }
        targets_unstacked = {
            'actions': [],
            'rewards_to_go': [],
        }
        target_dtypes = {
            'actions': jnp.int32,
            'rewards_to_go': jnp.float32,
        }

        # We need this to sample short trajectories less often
        trajectory_lengths = jnp.array(
            [len(t) for t in trajectory_batch.trajectories],
            dtype=jnp.int32)
        trajectory_logprobs = jnp.log(trajectory_lengths / jnp.sum(trajectory_lengths, dtype=jnp.float32))

        for example_index in range(model_batch_size):
            trajectory_index = jax.random.categorical(self._next_random_key(), trajectory_logprobs)
            trajectory = trajectory_batch.trajectories[trajectory_index]
            src_state_index = jax.random.randint(
                self._next_random_key(), shape=(), minval=0, maxval=len(trajectory) - 1)

            src_state = (
                trajectory.initial_state if src_state_index == 0
                else trajectory.elements[src_state_index - 1].observation
            )
            observations_unstacked['glyphs'].append(src_state['glyphs'])

            targets_unstacked['actions'].append(trajectory.elements[src_state_index].action)
            targets_unstacked['rewards_to_go'].append(rewards_to_go[trajectory_index][src_state_index])

        observations = {
            key: jnp.stack(values, axis=0)
            for key, values in observations_unstacked.items()
        }
        targets = {
            key: jnp.array(values, dtype=target_dtypes[key])
            for key, values in targets_unstacked.items()
        }

        return observations, targets

    def _compute_loss(self, model_params, observations_batch, targets_batch, rng):
        log_action_probs, value = self._model.apply(
            model_params, observations_batch, deterministic=False, rng=rng)
        advantage = targets_batch['rewards_to_go'] - value
        weights = jnp.ones(shape=advantage.shape)
        policy_gradient_loss = rlax.policy_gradient_loss(log_action_probs, targets_batch['actions'], advantage, weights)
        value_function_loss = jnp.mean(rlax.l2_loss(value, targets_batch['rewards_to_go']))
        return policy_gradient_loss + self._config['value_function_loss_weight'] * value_function_loss

    def _next_random_key(self):
        self._random_key, subkey = jax.random.split(self._random_key)
        return subkey

    def _build_model(self, observations_batch):
        assert self._model is None

        self._model = NethackPerceiverModel(num_actions=self.action_space.n)

        self._model_params = self._model.init(
            self._next_random_key(), observations_batch, deterministic=False, rng=self._next_random_key())

        self._optimizer = optax.adam(learning_rate=self._config['lr'])
        self._optimizer_state = self._optimizer.init(self._model_params)

    def act(self, observation_batch):
        if self._model is None:
            self._build_model(observation_batch)

        # TODO: we need some form of exploration
        log_action_probs, value = self._model.apply(
            self._model_params, observation_batch, deterministic=True, rng=self._next_random_key())
        return jax.random.categorical(self._next_random_key(), log_action_probs)

    def train_on_batch(self, trajectory_batch):
        assert self._model is not None

        # TODO: replace pair sampling by recurrence
        observations_batch, targets_batch = self._build_model_batch(trajectory_batch)
        grad_func = jax.grad(self._compute_loss, argnums=0)
        grads = grad_func(self._model_params, observations_batch, targets_batch, rng=self._next_random_key())
        updates, self._optimizer_state = self._optimizer.update(grads, self._optimizer_state)
        self._model_params = optax.apply_updates(self._model_params, updates)
