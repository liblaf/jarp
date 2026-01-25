import types
from collections.abc import Callable
from typing import Any

import jax.tree_util as jtu


def flatten_method(obj: types.MethodType) -> tuple[tuple[object], Callable[..., Any]]:
    return (obj.__self__,), obj.__func__


def flatten_method_with_keys(
    obj: types.MethodType,
) -> tuple[tuple[tuple[Any, object]], Callable[..., Any]]:
    return ((jtu.GetAttrKey("__self__"), obj.__self__),), obj.__func__


def unflatten_method(
    aux: Callable[..., Any], children: tuple[object]
) -> types.MethodType:
    return types.MethodType(aux, children[0])


def register_pytree_method() -> None:
    if types.MethodType in jtu.default_registry:
        return
    jtu.register_pytree_node(
        types.MethodType, flatten_method, unflatten_method, flatten_method_with_keys
    )
