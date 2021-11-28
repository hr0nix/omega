from dataclasses import field
from typing import Optional, Dict

import flax.linen as nn
import jax.numpy as jnp
import jax.random

from .nethack_perceiver_model import PerceiverStateEncoder


class RNDNetwork(nn.Module):
    state_encoder_config: Dict = field(default_factory=dict)
    output_dim: int = 32
    deterministic: Optional[bool] = None

    def setup(self):
        self._state_encoder = PerceiverStateEncoder(**self.state_encoder_config)
        self._output_network = nn.Dense(self.output_dim)

    def __call__(self, current_state_batch, rng, deterministic=None):
        deterministic = nn.module.merge_param('deterministic', self.deterministic, deterministic)

        # Encode state
        rng, subkey = jax.random.split(rng, 2)
        memory = self._state_encoder(current_state_batch, deterministic=deterministic, rng=rng)

        # Summarize encoded state in a compact vector
        memory = jnp.reshape(memory, newshape=[memory.shape[0], -1])
        output = self._output_network(memory)

        return output


class RNDNetworkPair(nn.Module):
    rnd_network_config: Dict = field(default_factory=dict)
    deterministic: Optional[bool] = None

    def setup(self):
        self._random_network = RNDNetwork(**self.rnd_network_config)
        self._predictor_network = RNDNetwork(**self.rnd_network_config)

    def __call__(self, current_state_batch, rng, deterministic=None):
        deterministic = nn.module.merge_param('deterministic', self.deterministic, deterministic)

        subkey1, subkey2 = jax.random.split(rng)
        random_state = self._random_network(current_state_batch, rng=subkey1, deterministic=deterministic)
        random_state = jax.lax.stop_gradient(random_state)  # Don't train the random network
        predicted_state = self._predictor_network(current_state_batch, rng=subkey2, deterministic=deterministic)

        loss = 0.5 * (predicted_state - random_state) ** 2
        loss_per_example = jnp.mean(loss, axis=1)

        return loss_per_example