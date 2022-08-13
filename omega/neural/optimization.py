import jax
import jax.numpy as jnp


def clip_gradient_by_norm(grad, max_norm, return_norm=False):
    norm = jnp.linalg.norm(jax.tree_util.tree_leaves(jax.tree_map(jnp.linalg.norm, grad)))
    factor = jnp.minimum(1.0, max_norm / (norm + 1e-6))
    clipped_grad = jax.tree_util.tree_map((lambda x: x * factor), grad)
    return (clipped_grad, norm) if return_norm else clipped_grad
