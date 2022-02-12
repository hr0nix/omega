import jax
import jax.numpy as jnp

import numpy as np


def _select_op(pytree, result_device, cpu_op, gpu_op):
    if result_device == 'cpu':
        return cpu_op
    elif result_device == 'gpu':
        return gpu_op
    elif result_device is not None:
        raise ValueError(f'Unknown device: {result_device}')

    if is_cpu(pytree):
        return cpu_op
    elif is_gpu(pytree):
        return gpu_op
    else:
        raise ValueError(f'Given pytree has a mix of CPU and GPU tensors, specify result_device explicitly')


def copy(pytree):
    return jax.tree_map(lambda x: x, pytree)


def update(pytree, *updates):
    if not isinstance(pytree, dict):
        raise ValueError('Top-level node must be a dict!')

    result = copy(pytree)
    for update in updates:
        result.update(update)
    return result


def expand_dims(pytree, axis, result_device=None):
    expand_dims_op = _select_op(pytree, result_device, np.expand_dims, jnp.expand_dims)
    return jax.tree_map(lambda t: expand_dims_op(t, axis=axis), pytree)


def squeeze(pytree, axis, result_device=None):
    squeeze_op = _select_op(pytree, result_device, np.squeeze, jnp.squeeze)
    return jax.tree_map(lambda t: squeeze_op(t, axis=axis), pytree)


def mean(pytree, result_device=None):
    mean_op = _select_op(pytree, result_device, np.mean, jnp.mean)
    return jax.tree_map(lambda t: mean_op(t), pytree)


def get_axis_dim(pytree, axis):
    leaves = jax.tree_leaves(pytree)
    axis_dim = leaves[0].shape[axis]
    assert all(l.shape[axis] == axis_dim for l in leaves)
    return axis_dim


def slice_from_batch(batch, slice_idx):
    return jax.tree_map(lambda t: t[slice_idx, ...], batch)


def stack(pytrees, axis, result_device=None):
    stack_op = _select_op(pytrees[0], result_device, np.stack, jnp.stack)
    return jax.tree_map(
        lambda *leaves: stack_op(leaves, axis=axis),
        *pytrees,
    )


def concatenate(pytrees, axis, result_device=None):
    concatenate_op = _select_op(pytrees, result_device, np.concatenate, jnp.concatenate)
    return jax.tree_map(
        lambda *leaves: concatenate_op(leaves, axis=axis),
        *pytrees,
    )


def to_cpu(pytree):
    return jax.tree_map(lambda t: np.asarray(t), pytree)


def to_gpu(pytree):
    return jax.tree_map(lambda t: jnp.asarray(t), pytree)


def is_cpu(pytree):
    return all(isinstance(l, np.ndarray) for l in jax.tree_leaves(pytree))


def is_gpu(pytree):
    return all(isinstance(l, jnp.ndarray) for l in jax.tree_leaves(pytree))