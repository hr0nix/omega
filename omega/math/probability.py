import jax
import jax.numpy as jnp
import distrax


def entropy(logits):
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    probs = jnp.exp(log_probs)
    log_probs_masked = jnp.where(probs == 0.0, jnp.zeros_like(log_probs), log_probs)
    return -jnp.sum(probs * log_probs_masked, axis=-1)


def cross_entropy(labels, logits):
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    log_probs_masked = jnp.where(labels == 0.0, jnp.zeros_like(log_probs), log_probs)
    return -jnp.sum(labels * log_probs_masked, axis=-1)


# TODO: uncomment after distrax stops triggering NaN errors
# def entropy(logits):
#     return distrax.Categorical(logits=logits).entropy()
#
#
# def cross_entropy(labels, logits):
#     return distrax.Categorical(probs=labels).cross_entropy(
#         distrax.Categorical(logits=logits)
#     )
