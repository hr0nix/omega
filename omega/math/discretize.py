import chex
import jax.nn
import jax.numpy as jnp


def round_to_closest_canonic_value(values, canonic_values):
    chex.assert_rank(canonic_values, 1)

    values = jnp.expand_dims(values, 1)
    values_keys_diff = jnp.abs(values - canonic_values)
    best_canonic_value_indices = jnp.argmin(values_keys_diff, axis=-1)

    return canonic_values[best_canonic_value_indices]


def discretize(values, lookup):
    chex.assert_type(values, jnp.float32)

    values = round_to_closest_canonic_value(values, jnp.array(list(lookup.keys())))
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
