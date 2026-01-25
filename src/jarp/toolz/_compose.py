from collections.abc import Callable, Iterable
from typing import Any, Protocol

import jax.tree_util as jtu
import tlz
from liblaf import grapes


@grapes.wraps(tlz.compose_left)
def compose_left(*args, **kwargs) -> Any:
    return tlz.compose_left(*args, **kwargs)


class Compose(Protocol):
    @property
    def first(self) -> Callable[..., Any]: ...
    @property
    def funcs(self) -> Iterable[Callable[..., Any]]: ...


def _compose_flatten(obj: Compose) -> tuple[list[Callable[..., Any]], None]:
    return [obj.first, *obj.funcs], None


def _compose_unflatten(
    _aux: None, children: list[Callable[..., Any]]
) -> Callable[..., Any]:
    return tlz.compose(*children)


jtu.register_pytree_node(
    tlz.functoolz.Compose,  # pyright: ignore[reportAttributeAccessIssue]
    _compose_flatten,
    _compose_unflatten,  # pyright: ignore[reportArgumentType]
)
