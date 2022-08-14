from absl import logging
from functools import partial

import jax
from jax.experimental import checkify

from contextlib import contextmanager


_GLOBAL_CHECKIFY_ERRORS = None


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
def override_checkify_checks(errors):
    global _GLOBAL_CHECKIFY_ERRORS
    old_errors = _GLOBAL_CHECKIFY_ERRORS
    _GLOBAL_CHECKIFY_ERRORS = errors
    yield
    _GLOBAL_CHECKIFY_ERRORS = old_errors


@contextmanager
def checkify_all(enabled=True):
    if enabled:
        logging.info('Force enabling all checkify checks')
        with override_checkify_checks(checkify.all_checks):
            yield
    else:
        yield


def throws_on_checkify_error(func, checkify_errors=checkify.user_checks):
    """
    Enable checkify checks for a function and throw if there were any errors.
    """
    checkify_errors = _GLOBAL_CHECKIFY_ERRORS or checkify_errors
    checked_func = checkify.checkify(func, errors=checkify_errors)

    def result_func(*args, **kwargs):
        err, result = checked_func(*args, **kwargs)
        checkify.check_error(err)
        return result

    return result_func


def method_throws_on_checkify_error(func, checkify_errors=checkify.user_checks):
    """
    Enable checkify checks for a class method and throw if there were any errors.
    """
    def result_func(*args, **kwargs):
        self, *args_no_self = args

        @partial(throws_on_checkify_error, checkify_errors=checkify_errors)
        def inner_func_no_self(*inner_args, **inner_kwargs):
            return func(self, *inner_args, **inner_kwargs)

        return inner_func_no_self(*args_no_self, **kwargs)

    return result_func
