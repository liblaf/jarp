from __future__ import annotations

import enum
import functools
import warnings
from collections.abc import Callable
from typing import Any, Literal, TypedDict, Unpack, dataclass_transform, overload

import attrs
import jax.tree_util as jtu

from ._field_specifiers import array, auto, field, static
from ._register import register_fieldz


class PyTreeType(enum.StrEnum):
    """Choose how a class should participate in JAX PyTree flattening."""

    DATA = enum.auto()
    NONE = enum.auto()
    STATIC = enum.auto()

    @classmethod
    def _missing_(cls, value: object) -> Any:
        if isinstance(value, str):
            return cls._value2member_map_.get(value.lower())
        if value is True or value is None:
            return cls.DATA
        if value is False:
            return cls.NONE
        return None


class DefineOptions(TypedDict, total=False):
    these: dict[str, Any] | None
    repr: bool
    unsafe_hash: bool | None
    hash: bool | None
    init: bool
    slots: bool
    frozen: bool
    weakref_slot: bool
    str: bool
    auto_attribs: bool
    kw_only: bool
    cache_hash: bool
    auto_exc: bool
    eq: bool | None
    order: bool | None
    auto_detect: bool
    getstate_setstate: bool | None
    on_setattr: attrs._OnSetAttrArgType | None  # ty:ignore[invalid-type-form]
    field_transformer: attrs._FieldTransformer | None
    match_args: bool

    pytree: PyTreeType | Literal["data", "none", "static"] | bool | None


# TODO: remove `kw_only` once <https://github.com/astral-sh/ty/issues/3115> is fixed
@overload
@dataclass_transform(field_specifiers=(attrs.field, array, auto, field, static))
def define[T: type](
    cls: T, /, *, kw_only: bool = False, **kwargs: Unpack[DefineOptions]
) -> T: ...
@overload
@dataclass_transform(field_specifiers=(attrs.field, array, auto, field, static))
def define[T: type](
    cls: None = None, *, kw_only: bool = False, **kwargs: Unpack[DefineOptions]
) -> Callable[[T], T]: ...
def define[T: type](maybe_cls: T | None = None, **kwargs: Any) -> Any:
    """Define an ``attrs`` class and optionally register it as a PyTree.

    Args:
        maybe_cls: Class being decorated. When omitted, return a configured
            decorator.
        **kwargs: Options forwarded to [`attrs.define`][attrs.define], plus
            ``pytree`` to control JAX registration. ``pytree="data"``
            registers fields with ``fieldz`` semantics, ``"static"`` registers
            the whole instance as a static value, and ``"none"`` leaves the
            class unregistered.

    Returns:
        The decorated class or a class decorator.
    """
    if maybe_cls is None:
        return functools.partial(define, **kwargs)
    pytree: PyTreeType = PyTreeType(kwargs.pop("pytree", None))
    frozen: bool = kwargs.get("frozen", False)
    if pytree is PyTreeType.STATIC and not frozen:
        warnings.warn(
            "Defining a static class that is not frozen may lead to unexpected behavior.",
            stacklevel=2,
        )
    cls: T = attrs.define(maybe_cls, **kwargs)  # ty:ignore[invalid-assignment]
    match pytree:
        case PyTreeType.DATA:
            register_fieldz(cls)
        case PyTreeType.STATIC:
            jtu.register_static(cls)
    return cls


@overload
@dataclass_transform(
    frozen_default=True, field_specifiers=(attrs.field, array, auto, field, static)
)
def frozen[T: type](
    cls: T, /, *, kw_only: bool = False, **kwargs: Unpack[DefineOptions]
) -> T: ...
@overload
@dataclass_transform(
    frozen_default=True, field_specifiers=(attrs.field, array, auto, field, static)
)
def frozen[T: type](
    cls: None = None, /, *, kw_only: bool = False, **kwargs: Unpack[DefineOptions]
) -> Callable[[T], T]: ...
def frozen[T: type](maybe_cls: T | None = None, **kwargs: Any) -> Any:
    """Define a frozen ``attrs`` class and register it as a data PyTree.

    This is the common choice for immutable structures whose array fields
    should participate in JAX transformations.
    """
    _warnings_hide = True
    if maybe_cls is None:
        return functools.partial(frozen, **kwargs)
    kwargs.setdefault("frozen", True)
    return define(maybe_cls, **kwargs)


@overload
@dataclass_transform(
    frozen_default=True, field_specifiers=(attrs.field, array, auto, field, static)
)
def frozen_static[T: type](
    cls: T, /, *, kw_only: bool = False, **kwargs: Unpack[DefineOptions]
) -> T: ...
@overload
@dataclass_transform(
    frozen_default=True, field_specifiers=(attrs.field, array, auto, field, static)
)
def frozen_static[T: type](
    cls: None = None, /, *, kw_only: bool = False, **kwargs: Unpack[DefineOptions]
) -> Callable[[T], T]: ...
def frozen_static[T: type](maybe_cls: T | None = None, **kwargs: Any) -> Any:
    """Define a frozen ``attrs`` class and register it as a static PyTree.

    Use this for immutable helper objects that should be treated as static
    metadata instead of flattening into JAX leaves.
    """
    _warnings_hide = True
    if maybe_cls is None:
        return functools.partial(frozen_static, **kwargs)
    kwargs.setdefault("frozen", True)
    kwargs.setdefault("pytree", PyTreeType.STATIC)
    return define(maybe_cls, **kwargs)
