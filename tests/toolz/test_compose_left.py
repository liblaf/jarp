from collections.abc import Callable
from typing import Any

import jax

import jarp


def inc(x: int) -> int:
    return x + 1


def test_compose_left() -> None:
    f: Callable[[int], int] = jarp.toolz.compose_left(inc, inc, inc)
    assert f(0) == 3
    leaves: list[Any]
    treedef: Any
    leaves, treedef = jax.tree.flatten(f)
    assert leaves == [inc, inc, inc]
    f_recon: Callable[[int], int] = jax.tree.unflatten(treedef, leaves)
    assert f_recon(0) == 3
