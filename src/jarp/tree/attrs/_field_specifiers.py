from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, TypedDict, Unpack

import attrs
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from liblaf import grapes


class FieldOptions[T](TypedDict, total=False):
    default: T
    validator: attrs._ValidatorArgType[T] | None
    repr: attrs._ReprArgType
    hash: bool | None
    init: bool
    metadata: Mapping[Any, Any] | None
    converter: (
        attrs._ConverterType
        | list[attrs._ConverterType]
        | tuple[attrs._ConverterType, ...]
        | None
    )
    factory: Callable[[], T] | None
    kw_only: bool | None
    eq: attrs._EqOrderType | None
    order: attrs._EqOrderType | None
    on_setattr: attrs._OnSetAttrArgType | None
    alias: str | None
    type: type | None


def array(**kwargs: Unpack[FieldOptions[ArrayLike | None]]) -> Array:
    if "default" in kwargs and "factory" not in kwargs:
        # JAX arrays are not hashable, so we should use a default factory
        default: Array = jnp.asarray(kwargs.pop("default"))
        kwargs["factory"] = lambda: default
    return field(**kwargs)  # pyright: ignore[reportArgumentType, reportCallIssue]


@grapes.wraps(attrs.field)
def auto(**kwargs) -> Any:
    kwargs.setdefault("auto", True)
    return field(**kwargs)


@grapes.wraps(attrs.field)
def field(**kwargs) -> Any:
    auto: bool = kwargs.pop("auto", False)
    static: bool = kwargs.pop("static", False)
    assert not (auto and static), "Field cannot be both auto and static"
    if auto:
        kwargs["metadata"] = {"auto": auto, **kwargs.pop("metadata", {})}
    if static:
        kwargs["metadata"] = {"static": static, **kwargs.pop("metadata", {})}
    return attrs.field(**kwargs)


@grapes.wraps(attrs.field)
def static(**kwargs) -> Any:
    kwargs.setdefault("static", True)
    return field(**kwargs)
