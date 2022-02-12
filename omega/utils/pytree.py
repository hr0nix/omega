import jax
import jax.numpy as jnp

import numpy as np


def _select_op(pytree, cpu_op, gpu_op):
    return cpu_op if is_cpu(pytree) else gpu_op


def copy(pytree):
    return jax.tree_map(lambda x: x, pytree)


def update(pytree, *updates):
    if not isinstance(pytree, dict):
        raise ValueError('Top-level node must be a dict!')

    result = copy(pytree)
    for update in updates:
        result.update(update)
    return result


def expand_dims(pytree, axis):
    expand_dims_op = _select_op(pytree, np.expand_dims, jnp.expand_dims)
    return jax.tree_map(lambda t: expand_dims_op(t, axis=axis), pytree)


def squeeze(pytree, axis):
    squeeze_op = _select_op(pytree, np.squeeze, jnp.squeeze)
    return jax.tree_map(lambda t: squeeze_op(t, axis=axis), pytree)


def mean(pytree):
    mean_op = _select_op(pytree, np.mean, jnp.mean)
    return jax.tree_map(lambda t: mean_op(t), pytree)


def get_axis_dim(pytree, axis):
    leaves = jax.tree_leaves(pytree)
    axis_dim = leaves[0].shape[axis]
    assert all(l.shape[axis] == axis_dim for l in leaves)
    return axis_dim


def slice_from_batch(batch, slice_idx):
    return jax.tree_map(lambda t: t[slice_idx, ...], batch)


def stack(pytrees, axis):
    stack_op = _select_op(pytrees[0], np.stack, jnp.stack)
    return jax.tree_map(
        lambda *leaves: stack_op(leaves, axis=axis),
        *pytrees,
    )


def concatenate(pytrees, axis):
    concatenate_op = _select_op(pytrees[0], np.concatenate, jnp.concatenate)
    return jax.tree_map(
        lambda *leaves: concatenate_op(leaves, axis=axis),
        *pytrees,
    )


def to_cpu(pytree):
    return jax.tree_map(lambda t: np.asarray(t), pytree)


def is_cpu(pytree):
    return all(isinstance(l, np.ndarray) for l in jax.tree_leaves(pytree))