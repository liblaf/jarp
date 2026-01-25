import types
from collections.abc import Callable
from typing import Any

import jax.tree_util as jtu

from ._utils import in_registry


def _method_flatten(obj: types.MethodType) -> tuple[tuple[object], Callable[..., Any]]:
    return (obj.__self__,), obj.__func__


def _method_flatten_with_keys(
    obj: types.MethodType,
) -> tuple[tuple[tuple[Any, object]], Callable[..., Any]]:
    return ((jtu.GetAttrKey("__self__"), obj.__self__),), obj.__func__


def _method_unflatten(
    aux: Callable[..., Any], children: tuple[object]
) -> types.MethodType:
    return types.MethodType(aux, children[0])


def register_pytree_method() -> None:
    if in_registry(types.MethodType):
        return
    jtu.register_pytree_node(
        types.MethodType, _method_flatten, _method_unflatten, _method_flatten_with_keys
    )
