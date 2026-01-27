from __future__ import annotations

import enum
import functools
from collections.abc import Callable
from typing import Any, Literal, TypedDict, Unpack, dataclass_transform, overload

import attrs
import jax.tree_util as jtu
from liblaf import grapes

from ._field_specifiers import array, auto, field, static
from ._register import register_fieldz


class PyTreeType(enum.StrEnum):
    DATA = enum.auto()
    NONE = enum.auto()
    STATIC = enum.auto()

    @classmethod
    def _missing_(cls, value: object) -> Any:
        if isinstance(value, str):
            return cls(value.lower())
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
    on_setattr: attrs._OnSetAttrArgType | None
    field_transformer: attrs._FieldTransformer | None
    match_args: bool

    pytree: PyTreeType | Literal["data", "none", "static"] | bool | None


@overload
@dataclass_transform(field_specifiers=(attrs.field, array, auto, field, static))
def define[T: type](cls: T, /, **kwargs: Unpack[DefineOptions]) -> T: ...
@overload
@dataclass_transform(field_specifiers=(attrs.field, array, auto, field, static))
def define[T: type](**kwargs: Unpack[DefineOptions]) -> Callable[[T], T]: ...
def define[T: type](maybe_cls: T | None = None, **kwargs) -> T | Callable:
    _warnings_hide = True
    if maybe_cls is None:
        return functools.partial(define, **kwargs)
    pytree: PyTreeType = PyTreeType(kwargs.pop("pytree", None))
    frozen: bool = kwargs.get("frozen", False)
    if pytree is PyTreeType.STATIC and not frozen:
        grapes.warnings.warn(
            "Defining a static class that is not frozen may lead to unexpected behavior."
        )
    cls: T = grapes.attrs.define(maybe_cls, **kwargs)
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
def frozen[T: type](cls: T, /, **kwargs: Unpack[DefineOptions]) -> T: ...
@overload
@dataclass_transform(
    frozen_default=True, field_specifiers=(attrs.field, array, auto, field, static)
)
def frozen[T: type](**kwargs: Unpack[DefineOptions]) -> Callable[[T], T]: ...
def frozen[T: type](maybe_cls: T | None = None, **kwargs) -> T | Callable:
    _warnings_hide = True
    if maybe_cls is None:
        return functools.partial(frozen, **kwargs)
    kwargs.setdefault("frozen", True)
    return define(maybe_cls, **kwargs)


@overload
@dataclass_transform(
    frozen_default=True, field_specifiers=(attrs.field, array, auto, field, static)
)
def frozen_static[T: type](cls: T, /, **kwargs: Unpack[DefineOptions]) -> T: ...
@overload
@dataclass_transform(
    frozen_default=True, field_specifiers=(attrs.field, array, auto, field, static)
)
def frozen_static[T: type](**kwargs: Unpack[DefineOptions]) -> Callable[[T], T]: ...
def frozen_static[T: type](maybe_cls: T | None = None, **kwargs) -> T | Callable:
    _warnings_hide = True
    if maybe_cls is None:
        return functools.partial(frozen_static, **kwargs)
    kwargs.setdefault("frozen", True)
    kwargs.setdefault("pytree", PyTreeType.STATIC)
    return define(maybe_cls, **kwargs)
