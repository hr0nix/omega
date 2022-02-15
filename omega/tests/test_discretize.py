import pytest
import numpy as np
import jax.numpy as jnp

from omega.math import discretize, onehot, undiscretize_expected, round_to_closest_canonic_value


@pytest.fixture
def lookup():
    return {
        0.01: 0,
        0.1: 1,
        1.0: 2,
    }


def test_round_to_closest_canonic_value():
    np.testing.assert_array_equal(
        round_to_closest_canonic_value(jnp.array([0.0, 0.066, 0.04, 0.9, -10.0]), jnp.array([0.0, 0.1, 1.0, -1.0])),
        jnp.array([0.0, 0.1, 0.0, 1.0, -1.0])
    )

def test_discretize_exact(lookup):
    np.testing.assert_array_equal(
        discretize(jnp.array([0.1, 1.0, 0.01]), lookup),
        jnp.array([1, 2, 0])
    )


def test_discretize_approx(lookup):
    np.testing.assert_array_equal(
        discretize(jnp.array([0.09, 1.1, 0.0, 0.56, 0.54]), lookup),
        jnp.array([1, 2, 0, 2, 1])
    )


def test_undiscretize_expected(lookup):
    probs = jnp.array([0.5, 0.2, 0.3])
    log_probs = jnp.log(probs)
    assert undiscretize_expected(log_probs, lookup) == pytest.approx(0.01 * 0.5 + 0.1 * 0.2 + 1.0 * 0.3)


def test_onehot():
    np.testing.assert_array_equal(
        onehot(jnp.array([1, 3]), num_values=5),
        jnp.array([[0, 1, 0, 0, 0], [0, 0, 0, 1, 0]])
    )

