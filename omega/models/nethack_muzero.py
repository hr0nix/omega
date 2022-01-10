from dataclasses import field
from typing import Optional, Dict

import flax.linen as nn
import jax.numpy as jnp
import jax.random
import chex

from omega.neural import TransformerNet

from ..utils.pytree import add_batch_dim, remove_batch_dim
from .nethack_state_encoder import PerceiverNethackStateEncoder
from .base import ItemSelector, ItemPredictor


class NethackMuZeroModelBase(nn.Module):
    def latent_state_shape(self):
        raise NotImplementedError()

    def representation(self, observations, rng, deterministic=None):
        raise NotImplementedError()

    def dynamics(self, previous_latent_states, actions, rng, deterministic=None):
        raise NotImplementedError()

    def prediction(self, latent_states, rng, deterministic=None):
        raise NotImplementedError()


class NethackPerceiverMuZeroModel(NethackMuZeroModelBase):
    num_actions: int
    reward_dim: int
    state_encoder_config: Dict = field(default_factory=dict)
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
        self._action_embedder = nn.Embed(
            num_embeddings=self.num_actions, features=self._state_encoder.memory_dim,
            name='action_embedder')
        self._dynamics_transformer = TransformerNet(
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

    def representation(self, observation_trajectory, rng, deterministic=None):
        """
        Produces the representation of a trajectory of observations.
        """
        chex.assert_rank(observation_trajectory['glyphs'], 3)  # timestamp, width, height
        chex.assert_rank(rng, 1)

        deterministic = nn.module.merge_param('deterministic', self.deterministic, deterministic)

        # TODO: augment with recurrence over observations, for now we just use latest observation
        latent_trajectory =  self._state_encoder(observation_trajectory, deterministic=deterministic, rng=rng)

        chex.assert_rank(latent_trajectory, 3)
        return latent_trajectory

    def dynamics(self, previous_latent_state, action, rng, deterministic=None):
        """
        Produces the next latent state and reward if a given action is taken at a given state.
        Inputs are assumed to be non-batched.
        """
        chex.assert_rank([previous_latent_state, action, rng], [2, 0, 1])

        deterministic = nn.module.merge_param('deterministic', self.deterministic, deterministic)
        dynamics_function_key, reward_predictor_key = jax.random.split(rng)

        # TODO: rewrite layers so that they don't assume that the batch dimension is present
        previous_latent_state = add_batch_dim(previous_latent_state)
        action = add_batch_dim(action)

        action_embedding = self._action_embedder(action)
        action_embedding = jnp.expand_dims(action_embedding, axis=-2)
        previous_latent_state_with_action = action_embedding + previous_latent_state
        next_latent_state = self._dynamics_transformer(
            previous_latent_state_with_action, deterministic=deterministic, rng=dynamics_function_key)
        log_reward_probs = self._reward_predictor(
            previous_latent_state_with_action, deterministic=deterministic, rng=reward_predictor_key)[..., 0]
        log_reward_probs = jax.nn.log_softmax(log_reward_probs, axis=-1)

        chex.assert_rank([next_latent_state, log_reward_probs], [3, 2])
        chex.assert_axis_dimension(next_latent_state, 0, 1)
        chex.assert_axis_dimension(log_reward_probs, 0, 1)
        return remove_batch_dim(next_latent_state), remove_batch_dim(log_reward_probs)


    def prediction(self, latent_state, rng, deterministic=None):
        """
        Computes the policy and the value estimate for a single (non-batched) state.
        """

        chex.assert_rank([latent_state, rng], [2, 1])

        deterministic = nn.module.merge_param('deterministic', self.deterministic, deterministic)
        policy_network_key, value_predictor_key = jax.random.split(rng)

        latent_state = add_batch_dim(latent_state)

        all_actions = jnp.arange(0, self.num_actions, dtype=jnp.int32)
        all_action_embeddings = self._action_embedder(all_actions)
        all_action_embeddings = add_batch_dim(all_action_embeddings)
        log_action_probs = self._policy_network(
            all_action_embeddings, latent_state, deterministic=deterministic, rng=policy_network_key)

        value = self._value_predictor(latent_state, deterministic=deterministic, rng=value_predictor_key)[..., 0, 0]

        chex.assert_rank([log_action_probs, value], [2, 1])
        chex.assert_axis_dimension(log_action_probs, 0, 1)
        chex.assert_axis_dimension(value, 0, 1)
        return remove_batch_dim(log_action_probs), remove_batch_dim(value)
