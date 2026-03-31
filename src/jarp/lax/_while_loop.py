from collections.abc import Callable
from typing import Any

import jax

type BooleanNumeric = Any


def while_loop[T](
    cond_fun: Callable[[T], BooleanNumeric],
    body_fun: Callable[[T], T],
    init_val: T,
    *,
    jit: bool = True,
) -> T:
    """Run a loop with either ``jax.lax.while_loop`` or Python control flow.

    Args:
        cond_fun: Predicate evaluated on the loop state.
        body_fun: Function that produces the next loop state.
        init_val: Initial loop state.
        jit: When true, dispatch to :func:`jax.lax.while_loop`. When false, run
            an eager Python ``while`` loop with the same callbacks.

    Returns:
        The final loop state.
    """
    if jit:
        return jax.lax.while_loop(cond_fun, body_fun, init_val)
    val: T = init_val
    while cond_fun(val):
        val: T = body_fun(val)
    return val
