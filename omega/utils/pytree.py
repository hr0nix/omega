import functools

import jax
import jax.numpy as jnp
import jaxlib.xla_extension

import numpy as np


def _select_op(pytree, result_backend, numpy_op, jax_op):
    if result_backend == 'numpy':
        return numpy_op
    elif result_backend == 'jax':
        return jax_op
    elif result_backend is not None:
        raise ValueError(f'Unknown backend: {result_backend}')

    if is_numpy(pytree):
        return numpy_op
    elif is_jax(pytree):
        return jax_op
    else:
        raise ValueError(f'Given pytree has a mix of JAX and numpy tensors, specify result_backend explicitly')


def copy_structure(pytree):
    return jax.tree_map(lambda x: x, pytree)


def get_schema(pytree):
    return jax.tree_map(lambda x: type(x), pytree)


def restore_schema(pytree, schema):
    """
    Puts tensors back to their locations specified in schema.
    """
    def restore_schema_map(tensor, type):
        if type == jaxlib.xla_extension.DeviceArray:
            tensor = jnp.asarray(tensor)
        return tensor
    return jax.tree_map(restore_schema_map, pytree, schema)


def update(pytree, *updates):
    if not isinstance(pytree, dict):
        raise ValueError('Top-level node must be a dict!')

    result = copy_structure(pytree)
    for update in updates:
        result.update(update)
    return result


def remove_keys(pytree, keys):
    result = {}
    for k, v in pytree.items():
        if k not in keys:
            result[k] = v
    return result


def expand_dims(pytree, axis, result_backend=None):
    expand_dims_op = _select_op(pytree, result_backend, np.expand_dims, jnp.expand_dims)
    return jax.tree_map(lambda t: expand_dims_op(t, axis=axis), pytree)


def squeeze(pytree, axis, result_backend=None):
    squeeze_op = _select_op(pytree, result_backend, np.squeeze, jnp.squeeze)
    return jax.tree_map(lambda t: squeeze_op(t, axis=axis), pytree)


def mean(pytree, result_backend=None):
    mean_op = _select_op(pytree, result_backend, np.mean, jnp.mean)
    return jax.tree_map(lambda t: mean_op(t), pytree)


def array_mean(pytrees, result_backend=None):
    add_op = _select_op(pytrees, result_backend, np.add, jnp.add)
    return jax.tree_map(lambda *t: functools.reduce(add_op, t) / len(t), *pytrees)


def get_axis_dim(pytree, axis):
    leaves = jax.tree_leaves(pytree)
    axis_dim = leaves[0].shape[axis]
    assert all(l.shape[axis] == axis_dim for l in leaves)
    return axis_dim


# TODO: introduce a universal slicer
def batch_dim_slice(batch, slice_idx):
    return jax.tree_map(lambda t: t[slice_idx, ...], batch)


def timestamp_dim_slice(batch, slice_idx):
    return jax.tree_map(lambda t: t[:, slice_idx, ...], batch)


def stack(pytrees, axis, result_backend=None):
    stack_op = _select_op(pytrees[0], result_backend, np.stack, jnp.stack)
    return jax.tree_map(
        lambda *leaves: stack_op(leaves, axis=axis),
        *pytrees,
    )


def split(pytree, size, axis, result_backend=None):
    split_op = _select_op(pytree, result_backend, np.split, jnp.split)
    split_pytree = jax.tree_map(lambda t: split_op(t, size, axis=axis), pytree)
    split_pytree = squeeze(split_pytree, axis=axis, result_backend=result_backend)
    return jax.tree_transpose(
        inner_treedef=jax.tree_structure([_ for _ in range(size)]),
        outer_treedef=jax.tree_structure(pytree),
        pytree_to_transpose=split_pytree)


def concatenate(pytrees, axis, result_backend=None):
    concatenate_op = _select_op(pytrees, result_backend, np.concatenate, jnp.concatenate)
    return jax.tree_map(
        lambda *leaves: concatenate_op(leaves, axis=axis),
        *pytrees,
    )


def to_numpy(pytree):
    return jax.tree_map(lambda t: np.asarray(t), pytree)


def to_jax(pytree):
    return jax.tree_map(lambda t: jnp.asarray(t), pytree)


def is_numpy(pytree):
    return all(isinstance(l, np.ndarray) for l in jax.tree_leaves(pytree))


def is_jax(pytree):
    return all(isinstance(l, jnp.ndarray) for l in jax.tree_leaves(pytree))
