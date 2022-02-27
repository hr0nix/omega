import chex
import jax.nn
import jax.numpy as jnp


def round_to_closest_canonic_value(values, canonic_values):
    """
    Round each value to the closest canonic value.
    """
    chex.assert_rank(canonic_values, 1)

    values = jnp.expand_dims(values, axis=-1)
    values_keys_diff = jnp.abs(values - canonic_values)
    best_canonic_value_indices = jnp.argmin(values_keys_diff, axis=-1)

    return canonic_values[best_canonic_value_indices]


def discretize(values, lookup):
    """
    Given a tensor of values and a lookup table, returns a tensor of lookup table values corresponding to
    lookup keys closest to original values.
    """
    chex.assert_type(values, jnp.float32)

    values = round_to_closest_canonic_value(values, jnp.array(list(lookup.keys())))
    zeros = jnp.zeros_like(values, dtype=jnp.int32)
    result = zeros
    # TODO: this isn't too fast, can be done in linear time (vmap?)
    for key, value in lookup.items():
        result += jnp.where(values == key, jnp.ones_like(values, dtype=jnp.int32) * value, zeros)
    return result


def discretize_onehot(values, lookup):
    """
    Given a tensor of values and a lookup table, returns a tensor of lookup table values corresponding to
    lookup keys closest to original values in one-hot representation.
    """
    return onehot(discretize(values, lookup), num_values=len(lookup))


def undiscretize_expected(logits, lookup):
    """
    Given a probability distribution over lookup values, returns the expected value of the corresponding lookup keys.
    """
    probs = jax.nn.softmax(logits, axis=-1)
    result = jnp.zeros(shape=logits.shape[:-1], dtype=jnp.float32)
    for k, v in lookup.items():
        result += k * probs[..., v]
    return result


def onehot(values, num_values):
    """
    Converts the given tensor to one-hot representation.
    """
    chex.assert_type(values, jnp.int32)
    return jnp.eye(num_values)[values]
