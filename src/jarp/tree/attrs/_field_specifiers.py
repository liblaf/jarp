from __future__ import annotations

import enum
from collections.abc import Callable, Mapping
from typing import Any, TypedDict, Unpack

import attrs
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from liblaf import grapes


class FieldType(enum.StrEnum):
    AUTO = enum.auto()
    DATA = enum.auto()
    META = enum.auto()

    @classmethod
    def _missing_(cls, value: object) -> Any:
        if isinstance(value, str):
            return cls(value.lower())
        if value is True:
            return cls.META
        if value is False or value is None:
            return cls.DATA
        return None

    def __bool__(self) -> bool:
        match self:
            case FieldType.META:
                return True
            case FieldType.AUTO | FieldType.DATA:
                # for consistency with `jax.tree_util.register_dataclass`
                return False


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

    static: FieldType | bool | None


def array(**kwargs: Unpack[FieldOptions[ArrayLike | None]]) -> Array:
    if "default" in kwargs and "factory" not in kwargs:
        default: ArrayLike | None = kwargs.pop("default")  # pyright: ignore[reportAssignmentType]
        if default is not None:
            default = jnp.asarray(default)
        kwargs["factory"] = lambda: default
    return field(**kwargs)  # pyright: ignore[reportArgumentType, reportCallIssue]


@grapes.wraps(attrs.field)
def auto(**kwargs) -> Any:
    kwargs.setdefault("static", FieldType.AUTO)
    return field(**kwargs)


@grapes.wraps(attrs.field)
def field(**kwargs) -> Any:
    if "static" in kwargs:
        kwargs["metadata"] = {
            "static": kwargs.pop("static"),
            **(kwargs.get("metadata") or {}),
        }
    return attrs.field(**kwargs)


@grapes.wraps(attrs.field)
def static(**kwargs) -> Any:
    # for consistency with `jax.tree_util.register_dataclass`
    kwargs.setdefault("static", True)
    return field(**kwargs)
