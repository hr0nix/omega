import pytest

import jax
from jax.experimental import checkify

from omega.utils.jax import throws_on_checkify_error, checkify_func, checkify_method


def func_with_check(x):
    checkify.check(x > 0, 'x must be positive')
    return x


class ClassWithCheck:
    def method_with_check(self, x):
        checkify.check(x > 0, 'x must be positive')
        return x


def test_checkify_func():
    checked_f = checkify_func(func_with_check)

    err, val = checked_f(1)
    err.throw()
    assert val == 1

    err, val = checked_f(-1)
    with pytest.raises(ValueError):
        err.throw()


def test_checkify_method():
    instance = ClassWithCheck()
    checked_f = checkify_method(ClassWithCheck.method_with_check)

    err, val = checked_f(instance, 1)
    err.throw()
    assert val == 1

    err, val = checked_f(instance, -1)
    with pytest.raises(ValueError):
        err.throw()


def test_throws_on_checkify_error():
    checked_f = throws_on_checkify_error(checkify_func(func_with_check))
    assert checked_f(1) == 1

    with pytest.raises(ValueError):
        checked_f(-1)


def test_throws_on_checkify_error_with_jit():
    checked_f = throws_on_checkify_error(jax.jit(checkify_func(func_with_check)))
    assert checked_f(1) == 1

    with pytest.raises(ValueError):
        checked_f(-1)

