import pytest
import math

import jax.numpy as jnp

from omega.math.probability import entropy


def test_entropy():
    assert entropy(jnp.log(jnp.asarray([0.3, 0.7]))) == pytest.approx(-0.3 * math.log(0.3) - 0.7 * math.log(0.7))


def test_entropy_with_zero_probs():
    assert entropy(jnp.log(jnp.asarray([0.3, 0.7, 0.0]))) == pytest.approx(-0.3 * math.log(0.3) - 0.7 * math.log(0.7))
