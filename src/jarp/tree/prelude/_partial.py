# ruff: noqa: SLF001

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

import jax.tree_util as jtu
import wrapt

from jarp.tree._filters import is_data


class Partial[**P, T](wrapt.PartialCallableObjectProxy):
    __wrapped__: Callable[..., T]
    _self_args: tuple[Any, ...]
    _self_kwargs: dict[str, Any]

    def __init__(self, func: Callable[..., T], /, *args: Any, **kwargs: Any) -> None:
        super().__init__(func, *args, **kwargs)
        self._self_args = args
        self._self_kwargs = kwargs

    if TYPE_CHECKING:

        def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T: ...


def partial[T](func: Callable[..., T], /, *args: Any, **kwargs: Any) -> Partial[..., T]:
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
