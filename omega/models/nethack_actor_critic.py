from dataclasses import field
from typing import Optional, Dict

import flax.linen as nn
import jax.numpy as jnp
import jax.random

from omega.neural import CrossTransformerNet, DenseNet

from .nethack_state_encoder import PerceiverNethackStateEncoder
from .base import ItemSelector, ItemEmbedder


class NethackPerceiverActorCriticModel(nn.Module):
    num_actions: int
    state_encoder_config: Dict = field(default_factory=dict)
    output_attention_num_heads: int = 8
    transformer_dropout: float = 0.1
    transformer_fc_inner_dim: int = 256
    num_policy_network_heads: int = 2
    num_policy_network_blocks: int = 1
    num_inverse_dynamics_network_heads: int = 2
    num_inverse_dynamics_network_blocks: int = 1
    num_value_network_blocks: int = 2
    deterministic: Optional[bool] = None

    def setup(self):
        self._state_encoder = PerceiverNethackStateEncoder(
            **self.state_encoder_config,
            name='state_encoder',
        )
        self._output_embedder = ItemEmbedder(
            num_items=1,  # state value
            embedding_dim=self._state_encoder.memory_dim,
            name='output_embedder',
        )
        self._action_embedder = ItemEmbedder(
            num_items=self.num_actions,
            embedding_dim=self._state_encoder.memory_dim,
            name='action_embedder',
        )
        self._output_transformer = CrossTransformerNet(
            num_blocks=1,
            dim=self._state_encoder.memory_dim,
            fc_inner_dim=self.transformer_fc_inner_dim,
            num_heads=self.output_attention_num_heads,
            dropout_rate=self.transformer_dropout,
            deterministic=self.deterministic,
            name='output_transformer',
        )
        self._policy_network = ItemSelector(
            transformer_dim=self._state_encoder.memory_dim,
            transformer_num_blocks=self.num_policy_network_blocks,
            transformer_num_heads=self.num_policy_network_heads,
            transformer_fc_inner_dim=self.transformer_fc_inner_dim,
            transformer_dropout=self.transformer_dropout,
            deterministic=self.deterministic,
            name='policy_network',
        )
        self._inverse_dynamics_model = ItemSelector(
            transformer_dim=self._state_encoder.memory_dim,
            transformer_num_blocks=self.num_inverse_dynamics_network_blocks,
            transformer_num_heads=self.num_inverse_dynamics_network_heads,
            transformer_fc_inner_dim=self.transformer_fc_inner_dim,
            transformer_dropout=self.transformer_dropout,
            deterministic=self.deterministic,
            name='inverse_dynamics_model',
        )
        self._value_network = DenseNet(
            num_blocks=self.num_value_network_blocks, dim=self._state_encoder.memory_dim, output_dim=1,
            name='value_network',
        )

    def __call__(self, current_state, next_state, rng, deterministic=None):
        deterministic = nn.module.merge_param('deterministic', self.deterministic, deterministic)

        rng, subkey = jax.random.split(rng)
        memory = self._state_encoder(current_state, deterministic=deterministic, rng=subkey)
        batch_size = memory.shape[0]

        # Attend to latent memory from each regular output
        rng, subkey = jax.random.split(rng)
        output_embeddings = self._output_embedder(batch_size)
        output_embeddings = self._output_transformer(
            output_embeddings, memory, deterministic=deterministic, rng=subkey)

        # Compute state values
        state_value = self._value_network(output_embeddings[:, 0, :])
        state_value = jnp.squeeze(state_value, axis=-1)

        # Embed actions (for policy network and inverse dynamics model)
        action_embeddings = self._action_embedder(batch_size)

        # Compute action probs
        rng, subkey = jax.random.split(rng)
        log_action_probs = self._policy_network(
            action_embeddings, memory, deterministic=deterministic, rng=subkey)

        # Model inverse dynamics: predict the action that transitions into the next state
        rng, subkey1, subkey2 = jax.random.split(rng, 3)
        next_state_memory = self._state_encoder(next_state, deterministic=deterministic, rng=subkey1)
        # TODO: positional embeddings don't allow to distinguish this memory and next
        combined_memory = jnp.concatenate([memory, next_state_memory], axis=1)
        action_embeddings = jax.lax.stop_gradient(action_embeddings)  # Do not update action embeddings
        log_id_action_probs = self._inverse_dynamics_model(
            action_embeddings, combined_memory, deterministic=deterministic, rng=subkey2)

        return log_action_probs, log_id_action_probs, state_value
