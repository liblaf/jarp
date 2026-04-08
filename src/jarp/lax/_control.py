from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, cast

import jax
from jaxtyping import ArrayLike, ScalarLike

from jarp import utils

if TYPE_CHECKING:
    from _typeshed import IdentityFunction


type BooleanNumeric = ScalarLike


def _wraps(wrapped: Callable[..., Any]) -> IdentityFunction:
    def decorator[**P, T](fallback: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(wrapped)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            with utils.suppress_jax_errors():
                return wrapped(*args, **kwargs)
            return fallback(*args, **kwargs)

        return wrapper

    return decorator


@_wraps(jax.lax.cond)
def cond[*Ts, T](
    pred: ScalarLike,
    true_fun: Callable[[*Ts], T],
    false_fun: Callable[[*Ts], T],
    *operands: *Ts,
) -> T:
    """Choose between two branches, then retry eagerly if JAX rejects them.

    The wrapper first calls [`jax.lax.cond`][jax.lax.cond]. If that raises
    [`jax.errors.JAXTypeError`][jax.errors.JAXTypeError] or
    [`jax.errors.JAXIndexError`][jax.errors.JAXIndexError], it logs the
    exception and reruns the selected branch in plain Python.

    Args:
        pred: Scalar predicate. Python truthiness decides which branch runs on
            the fallback path.
        true_fun: Branch evaluated when ``pred`` is true.
        false_fun: Branch evaluated when ``pred`` is false.
        *operands: Positional operands forwarded to the selected branch.

    Returns:
        The value returned by the selected branch.
    """
    if pred:
        return true_fun(*operands)
    return false_fun(*operands)


@_wraps(jax.lax.fori_loop)
def fori_loop[T](
    lower: int, upper: int, body_fun: Callable[[int, T], T], init_val: T, **kwargs: Any
) -> T:
    """Run a counted loop, then retry in Python if JAX rejects the body.

    The wrapper first calls [`jax.lax.fori_loop`][jax.lax.fori_loop]. If that
    raises [`jax.errors.JAXTypeError`][jax.errors.JAXTypeError] or
    [`jax.errors.JAXIndexError`][jax.errors.JAXIndexError], it logs the
    exception and runs an ordinary Python ``for`` loop instead.

    Args:
        lower: Inclusive loop lower bound.
        upper: Exclusive loop upper bound.
        body_fun: Callback that receives the iteration index and current loop
            value, then returns the next loop value.
        init_val: Initial loop value.
        **kwargs: Extra keyword arguments forwarded to
            [`jax.lax.fori_loop`][jax.lax.fori_loop] on the first attempt.
            They are ignored on the Python fallback path.

    Returns:
        The final loop value.
    """
    del kwargs
    val: T = init_val
    for i in range(lower, upper):
        val: T = body_fun(i, val)
    return val


@_wraps(jax.lax.switch)
def switch[*Ts, T](
    index: ArrayLike, branches: Sequence[Callable[[*Ts], T]], *operands: *Ts
) -> T:
    """Choose one branch by index, then retry eagerly if JAX rejects it.

    The wrapper first calls [`jax.lax.switch`][jax.lax.switch]. If that raises
    [`jax.errors.JAXTypeError`][jax.errors.JAXTypeError] or
    [`jax.errors.JAXIndexError`][jax.errors.JAXIndexError], it logs the
    exception, clamps ``index`` into the valid range, and dispatches in plain
    Python.

    Args:
        index: Branch index. The fallback path clamps the value into the valid
            range before dispatch.
        branches: Candidate branch functions.
        *operands: Positional operands forwarded to the selected branch.

    Returns:
        The value returned by the selected branch.
    """
    index: int = min(max(0, cast("int", index)), len(branches) - 1)
    return branches[index](*operands)


@_wraps(jax.lax.while_loop)
def while_loop[T](
    cond_fun: Callable[[T], BooleanNumeric], body_fun: Callable[[T], T], init_val: T
) -> T:
    """Run a loop, then retry in Python if JAX rejects the callbacks.

    The wrapper first calls [`jax.lax.while_loop`][jax.lax.while_loop]. If
    that raises [`jax.errors.JAXTypeError`][jax.errors.JAXTypeError] or
    [`jax.errors.JAXIndexError`][jax.errors.JAXIndexError], it logs the
    exception and reruns the loop eagerly in Python.

    Args:
        cond_fun: Predicate evaluated on the loop state.
        body_fun: Function that produces the next loop state.
        init_val: Initial loop state.

    Returns:
        The final loop state.
    """
    val: T = init_val
    while cond_fun(val):
        val: T = body_fun(val)
    return val
