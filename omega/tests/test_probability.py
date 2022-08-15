import math

import chex
import jax.numpy as jnp

from omega.math.probability import entropy, cross_entropy


def test_entropy():
    logits = jnp.log(jnp.asarray([0.3, 0.7]))
    entropy_value = entropy(logits=logits)
    chex.assert_rank(entropy_value, 0)
    assert jnp.allclose(entropy_value, -0.7 * math.log(0.7) - 0.3 * math.log(0.3))


def test_entropy_with_zeros():
    logits = jnp.log(jnp.asarray([0.3, 0.7, 0.0]))
    entropy_value = entropy(logits=logits)
    chex.assert_rank(entropy_value, 0)
    assert jnp.allclose(entropy_value, -0.7 * math.log(0.7) - 0.3 * math.log(0.3))


def test_batch_entropy():
    logits = jnp.log(jnp.asarray([[0.3, 0.7], [0.4, 0.6]]))
    entropy_value = entropy(logits=logits)
    chex.assert_rank(entropy_value, 1)
    assert jnp.allclose(
        entropy_value,
        jnp.asarray([-0.7 * math.log(0.7) - 0.3 * math.log(0.3), -0.6 * math.log(0.6) - 0.4 * math.log(0.4)])
    )


def test_cross_entropy_1():
    labels = jnp.asarray([1.0, 0.0])
    logits = jnp.log(jnp.asarray([0.3, 0.7]))
    cross_entropy_value = cross_entropy(labels=labels, logits=logits)
    chex.assert_rank(cross_entropy_value, 0)
    assert jnp.allclose(cross_entropy_value, -math.log(0.3))


def test_cross_entropy_2():
    labels = jnp.asarray([0.4, 0.6])
    logits = jnp.log(jnp.asarray([0.3, 0.7]))
    cross_entropy_value = cross_entropy(labels=labels, logits=logits)
    chex.assert_rank(cross_entropy_value, 0)
    assert jnp.allclose(cross_entropy_value, -0.4 * math.log(0.3) - 0.6 * math.log(0.7))


def test_cross_entropy_with_zeros_1():
    labels = jnp.asarray([1.0, 0.0])
    logits = jnp.log(jnp.asarray([1.0, 0.0]))
    cross_entropy_value = cross_entropy(labels=labels, logits=logits)
    chex.assert_rank(cross_entropy_value, 0)
    assert jnp.allclose(cross_entropy_value, 0.0)


def test_cross_entropy_with_zeros_2():
    labels = jnp.asarray([1.0, 0.0])
    logits = jnp.log(jnp.asarray([0.0, 1.0]))
    cross_entropy_value = cross_entropy(labels=labels, logits=logits)
    chex.assert_rank(cross_entropy_value, 0)
    assert cross_entropy_value == jnp.inf


def test_batch_cross_entropy():
    labels = jnp.asarray([[1.0, 0.0], [0.5, 0.5]])
    logits = jnp.log(jnp.asarray([[1.0, 0.0], [0.3, 0.7]]))
    cross_entropy_value = cross_entropy(labels=labels, logits=logits)
    chex.assert_rank(cross_entropy_value, 1)
    assert jnp.allclose(
        cross_entropy_value,
        jnp.asarray([0.0, -0.5 * math.log(0.3) - 0.5 * math.log(0.7)])
    )
