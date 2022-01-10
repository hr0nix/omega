import jax
import jax.numpy as jnp


def copy_pytree(pytree):
    return jax.tree_map(lambda x: x, pytree)


def update_pytree(pytree, *updates):
    if not isinstance(pytree, dict):
        raise ValueError('Top-level node must be a dict!')

    result = copy_pytree(pytree)
    for update in updates:
        result.update(update)
    return result


def add_fake_dim(pytree, axis):
    return jax.tree_map(lambda t: jnp.expand_dims(t, axis=axis), pytree)


def add_batch_dim(pytree):
    return add_fake_dim(pytree, axis=0)


def remove_fake_dim(pytree, axis):
    return jax.tree_map(lambda t: jnp.squeeze(t, axis=axis), pytree)


def remove_batch_dim(pytree):
    return remove_fake_dim(pytree, axis=0)
