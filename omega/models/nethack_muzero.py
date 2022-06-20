from dataclasses import field
from typing import Optional, Dict

import flax.linen as nn
import jax.numpy as jnp
import jax.random
import chex

from omega.neural import CrossTransformerNet

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

    def chance_outcome_encoder(self, latent_state, deterministic=None):
        raise NotImplementedError()

    def representation(self, prev_memory, prev_action, observation, deterministic=None):
        raise NotImplementedError()

    def afterstate_dynamics(self, previous_latent_state, action, deterministic=None):
        raise NotImplementedError()

    def dynamics(self, latent_afterstate, chance_outcome, deterministic=None):
        raise NotImplementedError()

    def afterstate_prediction(self, latent_afterstate, deterministic=None):
        raise NotImplementedError()

    def prediction(self, latent_state, deterministic=None):
        raise NotImplementedError()


class NethackPerceiverMuZeroModel(NethackMuZeroModelBase):
    num_actions: int
    num_chance_outcomes: int
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
    action_outcome_predictor_config: Dict = field(default_factory=lambda: {
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
        self._chance_outcome_embedder = nn.Dense(
            # No bias is needed: inputs are one-hot, so we can map them to anything
            features=self._state_encoder.memory_dim, use_bias=False,
            name='chance_outcome_embedder')
        self._memory_aggregator = CrossTransformerNet(
            dim=self._state_encoder.memory_dim,
            **self.memory_aggregator_config,
            name='memory_aggregator')
        self._dynamics_transformer = CrossTransformerNet(
            dim=self._state_encoder.memory_dim,
            **self.dynamics_transformer_config,
            name='dynamics_transformer')
        self._afterstate_dynamics_transformer = CrossTransformerNet(
            dim=self._state_encoder.memory_dim,
            **self.dynamics_transformer_config,
            name='afterstate_dynamics_transformer')
        self._reward_predictor = ItemPredictor(
            num_outputs=self.reward_dim, transformer_dim=self._state_encoder.memory_dim,
            **self.scalar_predictor_config,
            name='reward_predictor')
        self._chance_outcome_predictor = ItemSelector(
            transformer_dim=self._state_encoder.memory_dim,
            **self.action_outcome_predictor_config,
            name='chance_outcome_predictor')
        self._chance_outcome_encoder = ItemPredictor(
            num_outputs=self.num_chance_outcomes, transformer_dim=self._state_encoder.memory_dim,
            **self.scalar_predictor_config,
            name='chance_outcome_encoder_net')
        self._value_predictor = ItemPredictor(
            num_outputs=1, transformer_dim=self._state_encoder.memory_dim,
            **self.scalar_predictor_config,
            name='value_predictor')
        self._afterstate_value_predictor = ItemPredictor(
            num_outputs=1, transformer_dim=self._state_encoder.memory_dim,
            **self.scalar_predictor_config,
            name='afterstate_value_predictor')
        self._policy_network = ItemSelector(
            transformer_dim=self._state_encoder.memory_dim,
            **self.action_outcome_predictor_config,
            name='policy_network')

    def latent_state_shape(self):
        return self._state_encoder.num_memory_units, self._state_encoder.memory_dim

    def memory_shape(self):
        return self._state_encoder.num_memory_units, self._state_encoder.memory_dim

    def initial_memory_state(self):
        memory_indices = jnp.arange(0, self._state_encoder.num_memory_units, dtype=jnp.int32)
        return self._initial_memory_embedder(memory_indices)

    def chance_outcome_encoder(self, latent_state, deterministic=None):
        """
        Produces the embedding of a chance outcome that led to the given state.
        """
        chex.assert_rank(latent_state, 2)

        deterministic = nn.module.merge_param('deterministic', self.deterministic, deterministic)

        latent_state = pytree.expand_dims(latent_state, axis=0)

        chance_outcome_embedding = self._chance_outcome_encoder(latent_state, deterministic=deterministic)

        chex.assert_rank(chance_outcome_embedding, 2)

        # TODO: remove me
        return jax.nn.one_hot(0, num_classes=self.num_chance_outcomes, dtype=jnp.float32)

        #return pytree.squeeze(chance_outcome_embedding, axis=0)


    def representation(self, prev_memory, prev_action, observation, deterministic=None):
        """
        Produces the representation of an observation.
        """
        chex.assert_rank(observation['glyphs'], 2)  # rows, cols

        deterministic = nn.module.merge_param('deterministic', self.deterministic, deterministic)

        # TODO: rewrite layers so that they don't assume that the batch dimension is present
        observation = pytree.expand_dims(observation, axis=0)
        prev_memory = pytree.expand_dims(prev_memory, axis=0)
        prev_action = pytree.expand_dims(prev_action, axis=0)

        latent_observation = self._state_encoder(observation, deterministic=deterministic)

        # Fuse prev action embedding with prev memory
        prev_action_embedding = self._action_embedder(prev_action)
        prev_action_embedding = jnp.expand_dims(prev_action_embedding, axis=-2)
        prev_memory_with_action = jnp.concatenate([prev_memory, prev_action_embedding], axis=-2)

        # Attend from latent observation to prev memory
        representation = self._memory_aggregator(
            latent_observation, prev_memory_with_action, deterministic=deterministic)

        chex.assert_rank(representation, 3)
        chex.assert_axis_dimension(representation, 0, 1)

        # Get rid of the no longer needed batch dimension
        representation = pytree.squeeze(representation, axis=0)

        # Also return the representation as the updated memory value
        return representation, representation

    def afterstate_dynamics(self, previous_latent_state, action, deterministic=None):
        """
        Produces the afterstate resulting from acting with a given action in a given state.
        Inputs are assumed to be non-batched.
        """
        chex.assert_rank([previous_latent_state, action], [2, 0])

        deterministic = nn.module.merge_param('deterministic', self.deterministic, deterministic)

        # TODO: rewrite layers so that they don't assume that the batch dimension is present
        previous_latent_state = pytree.expand_dims(previous_latent_state, axis=0)
        action = pytree.expand_dims(action, axis=0)

        action_embedding = self._action_embedder(action)
        action_embedding = jnp.expand_dims(action_embedding, axis=-2)
        previous_latent_state_with_action = jnp.concatenate([action_embedding, previous_latent_state], axis=-2)

        latent_afterstate = self._afterstate_dynamics_transformer(
            previous_latent_state, previous_latent_state_with_action, deterministic=deterministic)

        chex.assert_rank(latent_afterstate, 3)
        return pytree.squeeze(latent_afterstate, axis=0)

    def dynamics(self, latent_afterstate, chance_outcome_one_hot, deterministic=None):
        """
        Produces the next latent state and reward if a given chance outcome happens at a given afterstate.
        Inputs are assumed to be non-batched.
        """
        chex.assert_rank([latent_afterstate, chance_outcome_one_hot], [2, 1])

        deterministic = nn.module.merge_param('deterministic', self.deterministic, deterministic)

        # TODO: rewrite layers so that they don't assume that the batch dimension is present
        latent_afterstate = pytree.expand_dims(latent_afterstate, axis=0)
        chance_outcome_one_hot = pytree.expand_dims(chance_outcome_one_hot, axis=0)

        chance_outcome_embedding = self._chance_outcome_embedder(chance_outcome_one_hot)
        chance_outcome_embedding = jnp.expand_dims(chance_outcome_embedding, axis=-2)
        #latent_afterstate_with_chance_outcome = jnp.concatenate([chance_outcome_embedding, latent_afterstate], axis=-2)
        latent_afterstate_with_chance_outcome = chance_outcome_embedding

        next_latent_state = self._dynamics_transformer(
            latent_afterstate, latent_afterstate_with_chance_outcome, deterministic=deterministic)

        log_reward_probs = self._reward_predictor(
            latent_afterstate_with_chance_outcome, deterministic=deterministic)
        log_reward_probs = jax.nn.log_softmax(log_reward_probs, axis=-1)

        chex.assert_rank([next_latent_state, log_reward_probs], [3, 2])
        chex.assert_axis_dimension(next_latent_state, 0, 1)
        chex.assert_axis_dimension(log_reward_probs, 0, 1)
        return (
            pytree.squeeze(next_latent_state, axis=0),
            pytree.squeeze(log_reward_probs, axis=0),
        )

    def prediction(self, latent_state, deterministic=None):
        """
        Computes the policy and the value estimate for a single (non-batched) state.
        """

        chex.assert_rank(latent_state, 2)

        deterministic = nn.module.merge_param('deterministic', self.deterministic, deterministic)

        latent_state = pytree.expand_dims(latent_state, axis=0)

        all_actions = jnp.arange(0, self.num_actions, dtype=jnp.int32)
        all_action_embeddings = self._action_embedder(all_actions)
        all_action_embeddings = pytree.expand_dims(all_action_embeddings, axis=0)
        log_action_probs = self._policy_network(all_action_embeddings, latent_state, deterministic=deterministic)

        value = jnp.squeeze(
            self._value_predictor(latent_state, deterministic=deterministic),
            axis=-1
        )

        chex.assert_rank([log_action_probs, value], [2, 1])
        chex.assert_axis_dimension(log_action_probs, 0, 1)
        chex.assert_axis_dimension(value, 0, 1)
        return (
            pytree.squeeze(log_action_probs, axis=0),
            pytree.squeeze(value, axis=0)
        )

    def afterstate_prediction(self, latent_afterstate, deterministic=None):
        """
        Computes the chance outcome distribution and the value estimate for a single (non-batched) afterstate.
        """
        chex.assert_rank(latent_afterstate, 2)

        deterministic = nn.module.merge_param('deterministic', self.deterministic, deterministic)

        latent_afterstate = pytree.expand_dims(latent_afterstate, axis=0)

        all_chance_outcomes = jnp.arange(0, self.num_chance_outcomes, dtype=jnp.int32)
        all_chance_outcomes_onehot = jax.nn.one_hot(
            all_chance_outcomes, num_classes=self.num_chance_outcomes, dtype=jnp.float32)
        all_chance_outcome_embeddings = self._chance_outcome_embedder(all_chance_outcomes_onehot)
        all_chance_outcome_embeddings = pytree.expand_dims(all_chance_outcome_embeddings, axis=0)
        log_chance_outcome_probs = self._chance_outcome_predictor(
            all_chance_outcome_embeddings, latent_afterstate, deterministic=deterministic)

        afterstate_value = jnp.squeeze(
            self._afterstate_value_predictor(latent_afterstate, deterministic=deterministic),
            axis=-1,
        )

        chex.assert_rank([log_chance_outcome_probs, afterstate_value], [2, 1])
        chex.assert_axis_dimension(log_chance_outcome_probs, 0, 1)
        chex.assert_axis_dimension(afterstate_value, 0, 1)
        return (
            pytree.squeeze(log_chance_outcome_probs, axis=0),
            pytree.squeeze(afterstate_value, axis=0)
        )
