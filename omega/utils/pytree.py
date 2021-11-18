from copy import copy

import jax
import jax.numpy as jnp


def dict_update(pytree, upd):
    if not isinstance(pytree, dict):
        raise ValueError('Top-level node must be a dict!')

    result = copy(pytree)
    result.update(upd)
    return result


def stack(trees):
    assert len(trees) > 0
    leaves_list = []
    treedef = None
    for tree in trees:
        leaves, treedef = jax.tree_flatten(tree)
        leaves_list.append(leaves)

    grouped_leaves = zip(*leaves_list)
    result_leaves = [jnp.stack(l) for l in grouped_leaves]
    return treedef.unflatten(result_leaves)
