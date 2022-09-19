import math

import chex
import jax.numpy as jnp
import rlax

from omega.math.probability import (
    entropy, cross_entropy, aggregate_mixture_log_probs, categorical_mean_stddev, ensemble_mmean_mstddev,
)


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


def test_aggregate_mixture_log_probs():
    mixture_log_probs = jnp.log(jnp.asarray([
        [0.3, 0.7], [0.5, 0.5],
    ]))
    log_probs = aggregate_mixture_log_probs(mixture_log_probs, axis=0)
    chex.assert_shape(log_probs, (2,))
    assert jnp.allclose(log_probs, jnp.log(jnp.asarray([(0.3 + 0.5) * 0.5, (0.7 + 0.5) * 0.5])))


def test_ensemble_mmean_mstddev():
    mixture_log_probs = jnp.log(jnp.asarray([
        [0.3, 0.7], [0.5, 0.5],
    ]))
    support = jnp.array([-1.0, 1.0])
    mmean, mstddev = ensemble_mmean_mstddev(mixture_log_probs, support, axis=0)
    assert jnp.allclose(mmean, 0.2)
    assert jnp.allclose(mstddev, 0.2)


def test_ensemble_mmean_mstddev_same_means():
    mixture_log_probs = jnp.log(jnp.asarray([
        [0.4, 0.2, 0.4], [0.3, 0.4, 0.3],
    ]))
    support = jnp.array([-1.0, 0.0, 1.0])
    mmean, mstddev = ensemble_mmean_mstddev(mixture_log_probs, support, axis=0)
    assert jnp.allclose(mmean, 0.0)
    assert jnp.allclose(mstddev, 0.0)


def test_categorical_mean_stddev():
    support = jnp.asarray([-1.0, 0.0, 1.0], dtype=jnp.float32)
    log_probs = jnp.log(jnp.asarray([
        [0.3, 0.5, 0.2],
        [0.8, 0.2, 0.0],
    ]))
    mean, stddev = categorical_mean_stddev(log_probs, support)
    chex.assert_shape([mean, stddev], (2,))
    assert jnp.allclose(mean[0], -1.0 * 0.3 + 1.0 * 0.2)
    assert jnp.allclose(mean[1], -1.0 * 0.8)
    assert jnp.allclose(
        stddev[0],
        jnp.sqrt(0.3 * (mean[0] + 1.0) ** 2 + 0.5 * mean[0] ** 2 + 0.2 * (mean[0] - 1.0) ** 2)
    )
    assert jnp.allclose(
        stddev[1],
        jnp.sqrt(0.8 * (mean[1] + 1.0) ** 2 + 0.2 * mean[1] ** 2)
    )


