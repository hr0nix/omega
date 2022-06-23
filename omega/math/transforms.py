import jax.numpy as jnp


def log_transform(v):
    return jnp.sign(v) * jnp.log(1.0 + v)
