import jax.numpy as jnp

from omega.utils.tensor import pad_zeros, pad_one_hot, masked_mean


def test_pad_zeros_1d():
    tensor = jnp.asarray([1.0, 2.0, 3.0])
    padded_tensor = pad_zeros(tensor, size=2)
    assert jnp.allclose(padded_tensor, jnp.asarray([1.0, 2.0, 3.0, 0.0, 0.0]))


def test_pad_zeros_2d_1():
    tensor = jnp.asarray([[1.0, 2.0, 3.0]])
    padded_tensor = pad_zeros(tensor, size=1, axis=0)
    assert jnp.allclose(padded_tensor, jnp.asarray([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]))


def test_pad_zeros_2d_2():
    tensor = jnp.asarray([[1.0, 2.0, 3.0]])
    padded_tensor = pad_zeros(tensor, size=1, axis=1)
    assert jnp.allclose(padded_tensor, jnp.asarray([[1.0, 2.0, 3.0, 0.0]]))


def test_pad_one_hot():
    tensor = jnp.asarray([[0.0, 1.0], [1.0, 0.0]])
    padded_tensor = pad_one_hot(tensor, size=2, value=0)
    assert jnp.allclose(padded_tensor, jnp.asarray([[0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]))


def test_masked_mean_last_axis():
    tensor = jnp.asarray([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
    mask = jnp.asarray([[0.0, 1.0, 1.0]])
    assert jnp.allclose(masked_mean(tensor, mask, axis=-1), jnp.asarray([2.5, 3.5]))


def test_masked_mean_first_axis():
    tensor = jnp.asarray([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
    mask = jnp.asarray([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0]])
    assert jnp.allclose(masked_mean(tensor, mask, axis=0, keepdims=True), jnp.asarray([[2.0, 2.0, 3.5]]))


def test_masked_mean_all_axes():
    tensor = jnp.asarray([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
    mask = jnp.asarray([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0]])
    assert jnp.allclose(masked_mean(tensor, mask), jnp.asarray(2.75))


def test_masked_mean_nan():
    tensor = jnp.asarray([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
    mask = jnp.asarray(0.0)
    assert jnp.isnan(masked_mean(tensor, mask))
