import jax
import jax.numpy as jnp


def clip_gradient_by_norm(grad, max_norm):
    norm = jnp.linalg.norm(jax.tree_util.tree_leaves(jax.tree_map(jnp.linalg.norm, grad)))
    factor = jnp.minimum(1.0, max_norm / (norm + 1e-6))
    return jax.tree_map((lambda x: x * factor), grad)
