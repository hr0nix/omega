import flax
import flax.traverse_util

from . import pytree


def flatten_params(params):
    """
    Given a deep pytree of tensors, returns a stump (1-level tree) with tensor keys corresponding to paths
    in the original tree.
    """
    params = flax.core.unfreeze(params)
    flat_params = {
        '/'.join(k): v
        for k, v in flax.traverse_util.flatten_dict(params).items()
    }
    return flat_params


def unflatten_params(flat_params):
    """
    Given a pytree produced by flatten_params, converts it back to its original form.
    """
    params = flax.traverse_util.unflatten_dict({
        tuple(k.split('/')): v
        for k, v in flat_params.items()
    })
    params = flax.core.freeze(params)
    return params


def merge_params(*params):
    """
    Merges multiple pytrees. If the same tensor is present in multiple pytrees,
    it will be taken from the last parameter where it was available.
    """
    flat_params = [flatten_params(p) for p in params]
    merged_params = pytree.update(flat_params[0], *flat_params[1:])
    return unflatten_params(merged_params)
