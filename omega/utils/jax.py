from absl import logging

import jax

from contextlib import contextmanager


@contextmanager
def disable_jit_if_no_gpu():
    if all(d.device_kind == 'cpu' for d in jax.local_devices()):
        logging.info('Disable JIT because there is no GPU')
        with jax.disable_jit():
            yield
    else:
        yield
