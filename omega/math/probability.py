import jax.numpy as jnp
import jax.nn
import chex


def entropy(logits):
    """
    Computes the entropy of a given probability distribution, correctly handling zero probabilities.
    """
    chex.assert_rank(logits, 1)

    log_probs = jax.nn.log_softmax(logits)
    probs = jnp.exp(log_probs)
    non_negative_probs = jnp.maximum(probs, jnp.finfo(probs.dtype).eps)  # to avoid -inf times zero
    return -jnp.sum(jnp.where(probs > 0, non_negative_probs * log_probs, jnp.zeros_like(probs)))
