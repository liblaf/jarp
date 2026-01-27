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
    if jit:
        return jax.lax.while_loop(cond_fun, body_fun, init_val)
    val: T = init_val
    while cond_fun(val):
        val = body_fun(val)
    return val
