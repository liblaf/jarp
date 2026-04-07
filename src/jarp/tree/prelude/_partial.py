# ruff: noqa: SLF001

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

import jax.tree_util as jtu
import wrapt

from jarp.tree._filters import is_data


class Partial[**P, T](wrapt.PartialCallableObjectProxy):
    """Store a partially applied callable as a PyTree-aware proxy.

    Bound arguments and keyword arguments flatten as PyTree children, while the
    wrapped callable itself is partitioned between dynamic data and static
    metadata when needed.
    """

    __wrapped__: Callable[..., T]
    _self_args: tuple[Any, ...]
    _self_kwargs: dict[str, Any]

    def __init__(self, func: Callable[..., T], /, *args: Any, **kwargs: Any) -> None:
        """Create a proxy that records bound arguments for PyTree flattening."""
        super().__init__(func, *args, **kwargs)
        self._self_args = args
        self._self_kwargs = kwargs

    if TYPE_CHECKING:

        def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T: ...


def partial[T](func: Callable[..., T], /, *args: Any, **kwargs: Any) -> Partial[..., T]:
    """Partially apply a callable and keep the result compatible with JAX trees."""
    return Partial(func, *args, **kwargs)


def _partial_flatten(obj: Partial) -> tuple[tuple, tuple]:
    func = obj.__wrapped__
    func_data, func_meta = (func, None) if is_data(func) else (None, func)
    return (obj._self_args, obj._self_kwargs, func_data), (func_meta,)


def _partial_flatten_with_keys(obj: Partial) -> tuple[tuple, tuple]:
    func = obj.__wrapped__
    func_data, func_meta = (func, None) if is_data(func) else (None, func)
    return (
        (jtu.GetAttrKey("_self_args"), obj._self_args),
        (jtu.GetAttrKey("_self_kwargs"), obj._self_kwargs),
        (jtu.GetAttrKey("__wrapped__"), func_data),
    ), (func_meta,)


def _partial_unflatten(aux: Iterable[Any], children: Iterable[Any]) -> Partial:
    (func_meta,) = aux
    args, kwargs, func_data = children
    func = func_data if func_meta is None else func_meta
    return Partial(func, *args, **kwargs)


jtu.register_pytree_node(
    Partial, _partial_flatten, _partial_unflatten, _partial_flatten_with_keys
)
