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


def ensemble_mean_stddev(ensemble_log_probs, support):
    """
    Given a uniform mixture of categorical distributions, compute the mean and the standard deviation of the mixture.
    Axis -2 corresponds to mixture elements, axis -1 corresponds to categorical outcomes.
    Support maps the categorical outcomes to the actual elements they represent.
    """
    assert len(ensemble_log_probs.shape) >= 2
    assert len(support.shape) == 1
    chex.assert_equal_shape_suffix([ensemble_log_probs, support], 1)

    log_normalizer = jnp.log(ensemble_log_probs.shape[-2])
    aggregate_log_probs = jax.nn.logsumexp(ensemble_log_probs - log_normalizer, axis=-2)
    aggregate_probs = jnp.exp(aggregate_log_probs)
    mean = jnp.sum(aggregate_probs * support, axis=-1)
    mean_sqr = jnp.sum(aggregate_probs * support ** 2, axis=-1)
    stddev = jnp.sqrt(mean_sqr - mean ** 2)
    chex.assert_rank([mean, stddev], len(ensemble_log_probs.shape) - 2)

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
