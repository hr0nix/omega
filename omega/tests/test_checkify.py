import pytest

import jax
import jax.numpy as jnp
from jax.experimental import checkify

from omega.utils.jax import throws_on_checkify_error, method_throws_on_checkify_error


def func_with_check(x):
    checkify.check(x > 0, 'x must be positive')
    return x


class ClassWithCheck:
    def method_with_check(self, x):
        checkify.check(x > 0, 'x must be positive')
        return x


def test_checkify():
    checked_f = checkify.checkify(func_with_check)
    err, val = checked_f(1)
    err.throw()
    assert val == 1

    err, val = checked_f(-1)
    with pytest.raises(ValueError):
        err.throw()


def test_check_error():
    checked_f = checkify.checkify(func_with_check)

    def throwing_f(x):
        err, val = checked_f(x)
        checkify.check_error(err)
        return val

    assert throwing_f(1) == 1
    with pytest.raises(ValueError):
        throwing_f(-1)


def test_throws_on_checkify_error():
    checked_f = throws_on_checkify_error(func_with_check)
    assert checked_f(1) == 1

    with pytest.raises(ValueError):
        checked_f(-1)


@pytest.mark.xfail(reason='https://github.com/google/jax/issues/11904', run=False)
def test_checkify_on_class_method():
    instance = ClassWithCheck()
    checked_f = checkify.checkify(ClassWithCheck.method_with_check)
    err, val = checked_f(instance, 1)
    err.throw()
    assert val == 1

    err, val = checked_f(instance, -1)
    with pytest.raises(ValueError):
        err.throw()


def test_method_throws_on_checkify_error():
    instance = ClassWithCheck()
    checked_f = method_throws_on_checkify_error(ClassWithCheck.method_with_check)
    assert checked_f(instance, 1) == 1

    with pytest.raises(ValueError):
        checked_f(instance, -1)


def test_checkify_with_while():
    def func_to_check_with_loop(n, v):
        def while_condition(loop_state):
            counter, _ = loop_state
            return counter > 0

        def loop_body(loop_state):
            counter, value = loop_state
            checkify.check(value > 0, 'value must be positive')
            return counter - 1, value - 1

        _, result = jax.lax.while_loop(while_condition, loop_body, (n, v))
        return result

    checked_f = throws_on_checkify_error(func_to_check_with_loop)

    assert checked_f(2, 2) == 0
    with pytest.raises(ValueError):
        checked_f(2, 1)


@pytest.mark.xfail(reason='https://github.com/google/jax/issues/11905', run=False)
def test_checkify_with_batched_while():
    def func_to_check_with_loop(n, v):
        def while_condition(loop_state):
            counter, _ = loop_state
            return counter > 0

        def loop_body(loop_state):
            counter, value = loop_state
            checkify.check(value > 0, 'value must be positive')
            return counter - 1, value - 1

        _, result = jax.lax.while_loop(while_condition, loop_body, (n, v))
        return result

    batched_func = jax.vmap(func_to_check_with_loop)
    checked_f = checkify.checkify(batched_func)

    err, val = checked_f(jnp.asarray([1, 2, 3]), jnp.asarray([5, 2, 4]))
    err.throw()
    assert jnp.all(val == jnp.asarray([4, 0, 1]))
