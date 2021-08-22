from copy import copy


def dict_update(pytree, upd):
    if not isinstance(pytree, dict):
        raise ValueError('Top-level node must be a dict!')

    result = copy(pytree)
    result.update(upd)
    return result
