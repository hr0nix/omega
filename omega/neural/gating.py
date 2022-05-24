from typing import Any, Callable, Iterable

import jax
import jax.numpy as jnp
from flax import linen as nn


class GRUGate(nn.Module):
    """
    GRU gating layer as proposed in TransformerXL paper (https://arxiv.org/pdf/1910.06764.pdf),
    section 3.2
    """

    kernel_init: Callable[[Any, Iterable[int], Any], Any] = nn.initializers.lecun_normal()
    bias_init: Callable[[Any, Iterable[int], Any], Any] = nn.initializers.zeros

    @staticmethod
    def _mvp(m, v):
        return jax.lax.dot_general(v, m, dimension_numbers=(((v.ndim - 1,), (0,)), ((), ())))

    def _w_param(self, name, shape):
        return self.param(name, self.kernel_init, shape)

    def _b_param(self, name, shape):
        return self.param(name, self.bias_init, shape)

    @nn.compact
    def __call__(self, x, y):
        w_shape = (x.shape[-1], x.shape[-1], )
        b_shape = (x.shape[-1], )

        w_r = self._w_param('w_r', w_shape)
        u_r = self._w_param('u_r', w_shape)
        w_z = self._w_param('w_z', w_shape)
        u_z = self._w_param('u_z', w_shape)
        w_g = self._w_param('w_g', w_shape)
        u_g = self._w_param('u_g', w_shape)
        b_z = self._b_param('b_g', b_shape)

        r = nn.sigmoid(self._mvp(w_r, y) + self._mvp(u_r, x))
        z = nn.sigmoid(self._mvp(w_z, y) + self._mvp(u_z, x) - b_z)
        h = nn.tanh(self._mvp(w_g, y) + self._mvp(u_g, r * x))
        return (1.0 - z) * x + z * h
