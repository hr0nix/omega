import flax
import flax.traverse_util

from . import pytree


def flatten_params(params):
    params = flax.core.unfreeze(params)
    flat_params = {
        '/'.join(k): v
        for k, v in flax.traverse_util.flatten_dict(params).items()
    }
    return flat_params


def unflatten_params(flat_params):
    params = flax.traverse_util.unflatten_dict({
        tuple(k.split('/')): v
        for k, v in flat_params.items()
    })
    params = flax.core.freeze(params)
    return params


def merge_params(*params):
    flat_params = [flatten_params(p) for p in params]
    merged_params = pytree.update(flat_params[0], *flat_params[1:])
    return unflatten_params(merged_params)
