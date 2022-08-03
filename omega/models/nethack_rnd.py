from dataclasses import field
from typing import Optional, Dict

import flax.linen as nn
import jax.numpy as jnp
import jax.random

from .nethack_state_encoder import PerceiverNethackStateEncoder


class NethackRNDNetwork(nn.Module):
    state_encoder_config: Dict = field(default_factory=dict)
    output_dim: int = 32
    deterministic: Optional[bool] = None

    def setup(self):
        self._state_encoder = PerceiverNethackStateEncoder(
            **self.state_encoder_config,
            name='state_encoder',
        )
        self._output_network = nn.Dense(
            features=self.output_dim,
            name='output_network',
        )

    def __call__(self, current_state_batch, deterministic=None):
        deterministic = nn.module.merge_param('deterministic', self.deterministic, deterministic)

        # Encode state
        memory = self._state_encoder(current_state_batch, deterministic=deterministic)

        # Summarize encoded state in a compact vector
        memory = jnp.reshape(memory, newshape=[memory.shape[0], -1])
        output = self._output_network(memory)

        return output


class NethackRNDNetworkPair(nn.Module):
    rnd_network_config: Dict = field(default_factory=dict)
    deterministic: Optional[bool] = None

    def setup(self):
        self._random_network = NethackRNDNetwork(
            **self.rnd_network_config,
            name='random_network',
        )
        self._predictor_network = NethackRNDNetwork(
            **self.rnd_network_config,
            name='predictor_network',
        )

    def __call__(self, current_state_batch, deterministic=None):
        deterministic = nn.module.merge_param('deterministic', self.deterministic, deterministic)

        random_state = self._random_network(
            current_state_batch, deterministic=deterministic)
        random_state = jax.lax.stop_gradient(random_state)  # Don't train the random network
        predicted_state = self._predictor_network(
            current_state_batch, deterministic=deterministic)

        loss = 0.5 * (predicted_state - random_state) ** 2
        loss_per_example = jnp.mean(loss, axis=1)

        return loss_per_example
