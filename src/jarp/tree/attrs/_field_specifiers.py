from __future__ import annotations

import enum
from collections.abc import Callable, Mapping
from typing import Any, TypedDict, Unpack

import attrs
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from ._utils import _wraps


class FieldType(enum.StrEnum):
    """Describe how a field participates in PyTree flattening."""

    AUTO = enum.auto()
    DATA = enum.auto()
    META = enum.auto()

    @classmethod
    def _missing_(cls, value: object) -> enum.Enum | None:
        if isinstance(value, str):
            return cls._value2member_map_.get(value.lower())
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
    on_setattr: attrs._OnSetAttrArgType | None  # ty:ignore[invalid-type-form]
    alias: str | None
    type: type | None

    static: FieldType | bool | None


def array(**kwargs: Unpack[FieldOptions[ArrayLike | None]]) -> Array:
    """Create a data field whose default is normalized to a JAX array.

    When ``default`` is a concrete array-like value, ``array`` rewrites it into
    a factory so each instance receives its own array object.
    """
    if "default" in kwargs and "factory" not in kwargs:
        default: ArrayLike | None = kwargs["default"]
        if not (default is None or isinstance(default, attrs.Factory)):  # ty:ignore[invalid-argument-type]
            default: Array = jnp.asarray(default)
            kwargs.pop("default")
            kwargs["factory"] = lambda: default
    return field(**kwargs)


@_wraps(attrs.field)
def auto(**kwargs) -> Any:
    """Create a field whose PyTree role is chosen from the runtime value."""
    kwargs.setdefault("static", FieldType.AUTO)
    return field(**kwargs)


@_wraps(attrs.field)
def field(**kwargs) -> Any:
    """Create an ``attrs`` field using jarp's ``static`` metadata convention."""
    if "static" in kwargs:
        kwargs["metadata"] = {
            "static": kwargs.pop("static"),
            **(kwargs.get("metadata") or {}),
        }
    return attrs.field(**kwargs)


@_wraps(attrs.field)
def static(**kwargs) -> Any:
    """Create a field that is always treated as static metadata."""
    # for consistency with `jax.tree_util.register_dataclass`
    kwargs.setdefault("static", True)
    return field(**kwargs)
