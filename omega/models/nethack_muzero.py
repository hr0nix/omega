from dataclasses import field
from typing import Optional, Dict

import flax.linen as nn
import jax.numpy as jnp
import jax.random
import chex

from omega.neural import TransformerNet, CrossTransformerNet

from ..utils import pytree
from .nethack_state_encoder import PerceiverNethackStateEncoder
from .base import ItemSelector, ItemPredictor


class NethackMuZeroModelBase(nn.Module):
    def latent_state_shape(self):
        raise NotImplementedError()

    def memory_shape(self):
        raise NotImplementedError()

    def initial_memory_state(self):
        raise NotImplementedError()

    def representation(self, prev_memory, prev_action, observation, rng, deterministic=None):
        raise NotImplementedError()

    def dynamics(self, previous_latent_states, actions, rng, deterministic=None):
        raise NotImplementedError()

    def prediction(self, latent_states, rng, deterministic=None):
        raise NotImplementedError()


class NethackPerceiverMuZeroModel(NethackMuZeroModelBase):
    num_actions: int
    reward_dim: int
    state_encoder_config: Dict = field(default_factory=dict)
    memory_aggregator_config: Dict = field(default_factory=lambda: {
        'num_blocks': 2,
        'num_heads': 4,
        'fc_inner_dim': 256,
    })
    dynamics_transformer_config: Dict = field(default_factory=lambda: {
        'num_blocks': 2,
        'num_heads': 4,
        'fc_inner_dim': 256,
    })
    scalar_predictor_config: Dict = field(default_factory=lambda: {
        'transformer_num_blocks': 2,
        'transformer_num_heads': 2,
        'transformer_fc_inner_dim': 256,
    })
    policy_network_config: Dict = field(default_factory=lambda: {
        'transformer_num_blocks': 2,
        'transformer_num_heads': 2,
        'transformer_fc_inner_dim': 256,
    })
    deterministic: Optional[bool] = None

    def setup(self):
        self._state_encoder = PerceiverNethackStateEncoder(
            **self.state_encoder_config,
            name='state_encoder')
        self._initial_memory_embedder = nn.Embed(
            num_embeddings=self._state_encoder.num_memory_units, features=self._state_encoder.memory_dim,
            name='initial_memory_embedder'
        )
        self._action_embedder = nn.Embed(
            num_embeddings=self.num_actions, features=self._state_encoder.memory_dim,
            name='action_embedder')
        self._memory_aggregator = CrossTransformerNet(
            dim=self._state_encoder.memory_dim,
            **self.memory_aggregator_config,
            name='memory_aggregator')
        self._dynamics_transformer = CrossTransformerNet(
            dim=self._state_encoder.memory_dim,
            **self.dynamics_transformer_config,
            name='dynamics_transformer')
        self._reward_predictor = ItemPredictor(
            num_outputs=self.reward_dim, transformer_dim=self._state_encoder.memory_dim,
            **self.scalar_predictor_config,
            name='reward_predictor')
        self._value_predictor = ItemPredictor(
            num_outputs=1, transformer_dim=self._state_encoder.memory_dim,
            **self.scalar_predictor_config,
            name='value_predictor')
        self._policy_network = ItemSelector(
            transformer_dim=self._state_encoder.memory_dim,
            **self.policy_network_config,
            name='policy_network')

    def latent_state_shape(self):
        return self._state_encoder.num_memory_units, self._state_encoder.memory_dim

    def memory_shape(self):
        return self._state_encoder.num_memory_units, self._state_encoder.memory_dim

    def initial_memory_state(self):
        memory_indices = jnp.arange(0, self._state_encoder.num_memory_units, dtype=jnp.int32)
        return self._initial_memory_embedder(memory_indices)

    def representation(self, prev_memory, prev_action, observation, rng, deterministic=None):
        """
        Produces the representation of an observation.
        """
        chex.assert_rank(observation['glyphs'], 2)  # width, height
        chex.assert_rank(rng, 1)

        deterministic = nn.module.merge_param('deterministic', self.deterministic, deterministic)

        state_encoder_key, memory_aggregator_key = jax.random.split(rng, 2)

        observation_batch = pytree.expand_dims(observation, axis=0)  # State encoder expects a batch
        latent_observation_batch = self._state_encoder(
            observation_batch, deterministic=deterministic, rng=state_encoder_key)
        latent_observation = pytree.squeeze(latent_observation_batch, axis=0)

        # Fuse prev action embedding with prev memory
        prev_action_embedding = self._action_embedder(prev_action)
        prev_action_embedding = jnp.expand_dims(prev_action_embedding, axis=-2)
        prev_memory_with_action = jnp.concatenate([prev_memory, prev_action_embedding], axis=-2)

        # Attend from latent observation to prev memory
        representation = self._memory_aggregator(
            latent_observation, prev_memory_with_action, deterministic=deterministic, rng=memory_aggregator_key)

        chex.assert_rank(representation, 2)
        # Also return the representation as the updated memory value
        return representation, representation

    def dynamics(self, previous_latent_state, action, rng, deterministic=None):
        """
        Produces the next latent state and reward if a given action is taken at a given state.
        Inputs are assumed to be non-batched.
        """
        chex.assert_rank([previous_latent_state, action, rng], [2, 0, 1])

        deterministic = nn.module.merge_param('deterministic', self.deterministic, deterministic)
        dynamics_function_key, reward_predictor_key = jax.random.split(rng)

        # TODO: rewrite layers so that they don't assume that the batch dimension is present
        previous_latent_state = pytree.expand_dims(previous_latent_state, axis=0)
        action = pytree.expand_dims(action, axis=0)

        action_embedding = self._action_embedder(action)
        action_embedding = jnp.expand_dims(action_embedding, axis=-2)
        previous_latent_state_with_action = jnp.concatenate([action_embedding, previous_latent_state], axis=-2)

        next_latent_state = self._dynamics_transformer(
            previous_latent_state, previous_latent_state_with_action,
            deterministic=deterministic, rng=dynamics_function_key)

        log_reward_probs = self._reward_predictor(
            previous_latent_state_with_action, deterministic=deterministic, rng=reward_predictor_key)
        log_reward_probs = jax.nn.log_softmax(log_reward_probs, axis=-1)

        chex.assert_rank([next_latent_state, log_reward_probs], [3, 2])
        chex.assert_axis_dimension(next_latent_state, 0, 1)
        chex.assert_axis_dimension(log_reward_probs, 0, 1)
        return (
            pytree.squeeze(next_latent_state, axis=0),
            pytree.squeeze(log_reward_probs, axis=0),
        )

    def prediction(self, latent_state, rng, deterministic=None):
        """
        Computes the policy and the value estimate for a single (non-batched) state.
        """

        chex.assert_rank([latent_state, rng], [2, 1])

        deterministic = nn.module.merge_param('deterministic', self.deterministic, deterministic)
        policy_network_key, value_predictor_key = jax.random.split(rng)

        latent_state = pytree.expand_dims(latent_state, axis=0)

        all_actions = jnp.arange(0, self.num_actions, dtype=jnp.int32)
        all_action_embeddings = self._action_embedder(all_actions)
        all_action_embeddings = pytree.expand_dims(all_action_embeddings, axis=0)
        log_action_probs = self._policy_network(
            all_action_embeddings, latent_state, deterministic=deterministic, rng=policy_network_key)

        value = jnp.squeeze(
            self._value_predictor(latent_state, deterministic=deterministic, rng=value_predictor_key),
            axis=-1
        )

        chex.assert_rank([log_action_probs, value], [2, 1])
        chex.assert_axis_dimension(log_action_probs, 0, 1)
        chex.assert_axis_dimension(value, 0, 1)
        return (
            pytree.squeeze(log_action_probs, axis=0),
            pytree.squeeze(value, axis=0)
        )
