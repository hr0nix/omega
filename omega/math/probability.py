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
    return -jnp.mean(jnp.where(probs > 0, probs * log_probs, jnp.zeros_like(probs)))
