import chex
import jax
import jax.numpy as jnp


def entropy(logits):
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    probs = jnp.exp(log_probs)
    log_probs_masked = jnp.where(probs == 0.0, jnp.zeros_like(log_probs), log_probs)
    return -jnp.sum(probs * log_probs_masked, axis=-1)


def cross_entropy(labels, logits):
    chex.assert_equal_shape([labels, logits])
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    log_probs_masked = jnp.where(labels == 0.0, jnp.zeros_like(log_probs), log_probs)
    return -jnp.sum(labels * log_probs_masked, axis=-1)


def aggregate_mixture_log_probs(log_probs, axis=0):
    """
    Given a uniform mixture of categorical distributions, compute the log-probs of the mixture
    represented as a single categorical.
    @param log_probs: the log-probs corresponding to mixture components.
    @param axis: the mixture dimension of the input tensor.
    @return: the log-probs of the resulting categorical.
    """
    assert len(log_probs.shape) >= 2
    assert axis not in [-1, len(log_probs.shape) - 1], 'Ensemble axis must be different from logits axis'

    log_normalizer = jnp.log(log_probs.shape[axis])
    return jax.nn.logsumexp(log_probs - log_normalizer, axis=axis)


def ensemble_mmean_mstddev(log_probs, support, axis=0):
    """
    Computes mean of means and standard deviation of means for a categorical ensemble.
    @param log_probs: the log-probs corresponding to mixture components.
    @param support: an array mapping categorical distribution outcomes to the actual values they represent.
    @param axis: the mixture dimension of the input tensor.
    @return: the mean and the standard deviation of means.
    """
    assert len(log_probs.shape) >= 2
    assert len(support.shape) == 1
    chex.assert_equal_shape_suffix([log_probs, support], 1)
    assert axis not in [-1, len(log_probs.shape) - 1], 'Ensemble axis must be different from logits axis'

    probs = jnp.exp(log_probs)
    means = jnp.sum(probs * support, axis=-1)
    mean_of_means = jnp.mean(means, axis=axis)
    stddev_of_means = jnp.std(means, axis=axis)

    return mean_of_means, stddev_of_means


def categorical_mean_stddev(log_probs, support):
    """
    Computes mean and standard deviation of a categorical distribution.
    @param log_probs: the log-probs representing the distribution.
    @param support: an array mapping categorical distribution outcomes to the actual values they represent.
    @return: the mean and the standard deviation of the distribution.
    """
    assert len(log_probs.shape) >= 1
    assert len(support.shape) == 1
    chex.assert_equal_shape_suffix([log_probs, support], 1)

    probs = jnp.exp(log_probs)
    mean = jnp.sum(probs * support, axis=-1)
    mean_sqr = jnp.sum(probs * support ** 2, axis=-1)
    stddev = jnp.sqrt(mean_sqr - mean ** 2)
    chex.assert_rank([mean, stddev], len(log_probs.shape) - 1)

    return mean, stddev


# TODO: uncomment after distrax stops triggering NaN errors
# def entropy(logits):
#     return distrax.Categorical(logits=logits).entropy()
#
#
# def cross_entropy(labels, logits):
#     return distrax.Categorical(probs=labels).cross_entropy(
#         distrax.Categorical(logits=logits)
#     )
