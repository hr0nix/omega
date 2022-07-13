from absl import logging

import jax

from contextlib import contextmanager


@contextmanager
def conditionally_disable_jit(should_disable):
    if should_disable:
        logging.info('JIT disabled')
        with jax.disable_jit():
            yield
    else:
        yield


@contextmanager
def disable_jit_if_no_gpu():
    should_disable = all(d.device_kind == 'cpu' for d in jax.local_devices())
    with conditionally_disable_jit(should_disable):
        yield
