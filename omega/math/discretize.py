import chex
import jax.nn
import jax.numpy as jnp


def discretize(values, lookup):
    chex.assert_type(values, jnp.float32)

    zeros = jnp.zeros_like(values, dtype=jnp.int32)
    result = zeros
    # TODO: this isn't too fast, can be done in linear time (vmap?)
    for key, value in lookup.items():
        result += jnp.where(values == key, jnp.ones_like(values, dtype=jnp.int32) * value, zeros)
    return result


def discretize_onehot(values, lookup):
    return onehot(discretize(values, lookup), num_values=len(lookup))


def undiscretize_expected(logits, lookup):
    probs = jax.nn.softmax(logits, axis=-1)
    result = jnp.zeros(shape=logits.shape[:-1], dtype=jnp.float32)
    for k, v in lookup.items():
        result += k * probs[..., v]
    return result


def onehot(values, num_values):
    return jnp.eye(num_values)[values]
