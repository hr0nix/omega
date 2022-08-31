from absl import logging
from functools import partial

import jax
from jax.experimental import checkify

from contextlib import contextmanager


_FORCE_CHECKIFY_CHECKS = None


@contextmanager
def disable_jit(should_disable=True):
    if should_disable:
        logging.info('JIT disabled')
        with jax.disable_jit():
            yield
    else:
        yield


@contextmanager
def disable_jit_if_no_gpu():
    should_disable = all(d.device_kind == 'cpu' for d in jax.local_devices())
    with disable_jit(should_disable):
        yield


@contextmanager
def force_checkify_checks(errors):
    global _FORCE_CHECKIFY_CHECKS
    old_errors = _FORCE_CHECKIFY_CHECKS
    _FORCE_CHECKIFY_CHECKS = errors
    yield
    _FORCE_CHECKIFY_CHECKS = old_errors


def checkify_func(func, errors=None):
    """
    Enable checkify checks for a function. If errors are not specified,
    globally forced checks will be enabled. If no globally forced checks are set, only user checks will be enabled.
    """
    errors = errors or _FORCE_CHECKIFY_CHECKS or checkify.user_checks
    return checkify.checkify(func, errors=errors)


def checkify_method(func, errors=None):
    """
    Analogue of checkify_func, but for class methods.
    TODO: remove this function after checkify_func starts working for class methods,
    TODO: see https://github.com/google/jax/issues/11904
    """
    def result_func(*args, **kwargs):
        self, *args_no_self = args

        @partial(checkify_func, errors=errors)
        def inner_func_no_self(*inner_args, **inner_kwargs):
            return func(self, *inner_args, **inner_kwargs)

        return inner_func_no_self(*args_no_self, **kwargs)

    return result_func


def throws_on_checkify_error(func):
    """
    Throws if a checkified function returns an error. If used with jit, the order of transformations
    should be throws_on_checkify_error(jit(checkify_func(f))).
    """
    def result_func(*args, **kwargs):
        err, result = func(*args, **kwargs)
        checkify.check_error(err)
        return result

    return result_func