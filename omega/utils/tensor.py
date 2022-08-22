import chex
import jax
import jax.numpy as jnp


def pad_values(tensor, value, size, axis=-1):
    padding_shape = list(tensor.shape)
    padding_shape[axis] = size
    padding = jnp.full(padding_shape, value, dtype=tensor.dtype)
    return jnp.concatenate([tensor, padding], axis=axis)


def pad_zeros(tensor, size, axis=-1):
    return pad_values(tensor, value=0, size=size, axis=axis)


def pad_one_hot(tensor, size, value):
    assert len(tensor.shape) >= 2
    chex.assert_axis_dimension_gt(tensor, axis=-1, val=value)

    padding = jax.nn.one_hot(
        jnp.full(size, value, dtype=jnp.int32),
        num_classes=tensor.shape[-1],
        dtype=tensor.dtype,
    )
    return jnp.concatenate([tensor, padding], axis=-2)


def tile_along_new_axis(tensor, axis, num_reps):
    tensor = jnp.expand_dims(tensor, axis=axis)
    return jnp.repeat(tensor, num_reps, axis=axis)


def masked_mean(tensor, mask, axis=None, keepdims=False, allow_zero_mask=False):
    chex.assert_is_broadcastable(mask.shape, tensor.shape)
    chex.assert_type(mask, jnp.bool_)

    # To avoid division by zero in case of zero mask
    eps = jnp.finfo(jnp.float32).tiny if allow_zero_mask else 0.0
    normalizer = 1.0 / (jnp.sum(mask, axis=axis, keepdims=True, dtype=jnp.float32) + eps)

    result = jnp.sum(tensor * mask, axis=axis, keepdims=True) * normalizer
    if not keepdims:
        result = jnp.squeeze(result, axis=axis)

    return result


