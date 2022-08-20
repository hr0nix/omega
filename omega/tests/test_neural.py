import chex
import jax
import jax.numpy as jnp
import flax.linen as nn

from omega.neural.ensemble import DropoutEnsemble


class DenseWithDropout(nn.Module):
    output_dim: int

    @nn.compact
    def __call__(self, inputs):
        x = nn.Dense(features=self.output_dim)(inputs)
        x = nn.relu(x)
        x = nn.Dropout(rate=0.5)(x, deterministic=False)
        return x


class DenseWithDropoutEnsembleNet(nn.Module):
    output_dim: int

    def setup(self):
        self._net = DropoutEnsemble(
            element_type=DenseWithDropout,
            element_config={'output_dim': self.output_dim},
            name='dropout_ensemble',
        )

    def __call__(self, inputs, ensemble_size):
        return self._net(inputs, ensemble_size)


def test_dropout_ensemble():
    model = DenseWithDropoutEnsembleNet(output_dim=5)
    rng = jax.random.PRNGKey(31337)
    params = model.init({'dropout': rng, 'params': rng}, jnp.zeros((1, 10)), ensemble_size=1)
    outputs = model.apply(params, jnp.zeros((1, 10)), ensemble_size=3, rngs={'dropout': rng})
    chex.assert_shape(outputs, (3, 1, 5))
