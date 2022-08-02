import jax
from flax import linen as nn


class Gate(nn.Module):
    """
    A convenience class for selecting the desired gate type.
    """
    type: str

    @nn.compact
    def __call__(self, x, y):
        if self.type == 'skip_connection':
            return SkipConnectionGate(name='skip')(x, y)
        elif self.type == 'gru':
            return GRUGate(name='gru')(x, y)
        elif self.type == 'output':
            return OutputGate(name='output')(x, y)
        elif self.type == 'highway':
            return HighwayGate(name='highway')(x, y)
        else:
            raise ValueError('Unknown gate type: {}'.format(self.type))


class SkipConnectionGate(nn.Module):
    """
    A regular skip-connection.
    """
    @nn.compact
    def __call__(self, x, y):
        return x + y


class OutputGate(nn.Module):
    """
    Output gating layer as proposed in "Stabilizing Transformers for Reinforcement Learning" paper
    (https://arxiv.org/pdf/1910.06764.pdf), section 3.2
    """
    @nn.compact
    def __call__(self, x, y):
        modulation = nn.sigmoid(nn.Dense(features=x.shape[-1], name='modulation')(x))
        return x + modulation * y


class HighwayGate(nn.Module):
    """
    Highway gating layer as proposed in "Highway Networks" paper
    (https://arxiv.org/pdf/1505.00387.pdf)
    """

    bias_init: float = 3.0  # Initialize to pass information through the gate by default

    @nn.compact
    def __call__(self, x, y):
        dense = nn.Dense(features=x.shape[-1], name='modulation', bias_init=nn.initializers.constant(self.bias_init))
        modulation = nn.sigmoid(dense(x))
        return modulation * x + (1 - modulation) * y


class GRUGate(nn.Module):
    """
    GRU gating layer as proposed in "Stabilizing Transformers for Reinforcement Learning" paper
    (https://arxiv.org/pdf/1910.06764.pdf), section 3.2
    """

    bias_init: float = 3.0  # Initialize to pass information through the gate by default

    @staticmethod
    def _mvp(m, v):
        return jax.lax.dot_general(v, m, dimension_numbers=(((v.ndim - 1,), (0,)), ((), ())))

    def _w_param(self, name, shape):
        return self.param(name, nn.initializers.lecun_normal(), shape)

    def _b_param(self, name, shape):
        return self.param(name, nn.initializers.constant(self.bias_init), shape)

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
