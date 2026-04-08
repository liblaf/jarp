import logging
from collections.abc import Callable, Sequence
from typing import Any, cast

import jax
from jaxtyping import Array, ArrayLike, ScalarLike

from jarp import utils

type BooleanNumeric = ScalarLike

logger: logging.Logger = logging.getLogger(__name__)


def cond[*Ts, T](
    pred: ScalarLike,
    true_fun: Callable[[*Ts], T],
    false_fun: Callable[[*Ts], T],
    *operands: *Ts,
) -> T:
    """Choose between two branches with optional eager execution.

    Args:
        pred: Scalar predicate. When `jit=False`, Python truthiness decides
            which branch runs.
        true_fun: Branch evaluated when ``pred`` is true.
        false_fun: Branch evaluated when ``pred`` is false.
        *operands: Positional operands forwarded to the selected branch.
        jit: When true, dispatch to [`jax.lax.cond`][jax.lax.cond]. When false,
            execute the selected branch directly in Python.

    Returns:
        The value returned by the selected branch.
    """
    with utils.suppress_jax_errors():
        return jax.lax.cond(pred, true_fun, false_fun, *operands)
    if pred:
        return true_fun(*operands)
    return false_fun(*operands)


def fori_loop[T](
    lower: int, upper: int, body_fun: Callable[[int, T], T], init_val: T, **kwargs: Any
) -> T:
    """Run a counted loop with either JAX or Python control flow.

    Args:
        lower: Inclusive loop lower bound.
        upper: Exclusive loop upper bound.
        body_fun: Callback that receives the iteration index and current loop
            value, then returns the next loop value.
        init_val: Initial loop value.
        jit: When true, dispatch to [`jax.lax.fori_loop`][jax.lax.fori_loop].
            When false, run a Python ``for`` loop.
        **kwargs: Extra keyword arguments forwarded to
            [`jax.lax.fori_loop`][jax.lax.fori_loop] when ``jit=True``.

    Returns:
        The final loop value.
    """
    with utils.suppress_jax_errors():
        return jax.lax.fori_loop(lower, upper, body_fun, init_val, **kwargs)
    val: T = init_val
    for i in range(lower, upper):
        val: T = body_fun(i, val)
    return val


def switch[*Ts, T](
    index: ArrayLike, branches: Sequence[Callable[[*Ts], T]], *operands: *Ts
) -> T:
    """Choose one branch by index with optional eager execution.

    Args:
        index: Branch index. When ``jit=False``, the value is clamped into the
            valid range before dispatch.
        branches: Candidate branch functions.
        *operands: Positional operands forwarded to the selected branch.
        jit: When true, dispatch to [`jax.lax.switch`][jax.lax.switch]. When
            false, execute the selected branch directly in Python.

    Returns:
        The value returned by the selected branch.
    """
    with utils.suppress_jax_errors():
        return jax.lax.switch(index, branches, *operands)
    index: int = min(max(0, cast("int", index)), len(branches) - 1)
    return branches[index](*operands)


def while_loop[T](
    cond_fun: Callable[[T], BooleanNumeric], body_fun: Callable[[T], T], init_val: T
) -> T:
    """Run a loop with either ``jax.lax.while_loop`` or Python control flow.

    Args:
        cond_fun: Predicate evaluated on the loop state.
        body_fun: Function that produces the next loop state.
        init_val: Initial loop state.
        jit: When true, dispatch to
            [`jax.lax.while_loop`][jax.lax.while_loop]. When false, run an
            eager Python ``while`` loop with the same callbacks.

    Returns:
        The final loop state.
    """
    with utils.suppress_jax_errors():
        return jax.lax.while_loop(cond_fun, body_fun, init_val)
    val: T = init_val
    while cond_fun(val):
        val: T = body_fun(val)
    return val
