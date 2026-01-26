from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, TypedDict, Unpack, dataclass_transform, overload

import attrs
import jax.tree_util as jtu
from liblaf import grapes

from ._field_specifiers import array, auto, field, static
from ._register import register_fieldz


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

    static: bool


@overload
@dataclass_transform(field_specifiers=(attrs.field, array, auto, field, static))
def define[T: type](cls: T, /, **kwargs: Unpack[DefineOptions]) -> T: ...
@overload
@dataclass_transform(field_specifiers=(attrs.field, array, field, static))
def define[T: type](**kwargs: Unpack[DefineOptions]) -> Callable[[T], T]: ...
def define[T: type](maybe_cls: T | None = None, **kwargs) -> T | Callable:
    _warnings_hide = True
    if maybe_cls is None:
        return functools.partial(define, **kwargs)
    static: bool = kwargs.pop("static", False)
    if static and not kwargs.get("frozen", False):
        grapes.warnings.warn(
            "Defining a static class that is not frozen may lead to unexpected behavior."
        )
    cls: T = grapes.attrs.define(maybe_cls, **kwargs)
    if static:
        jtu.register_static(cls)
    else:
        register_fieldz(cls)
    return cls


@overload
@dataclass_transform(
    frozen_default=True, field_specifiers=(attrs.field, array, auto, field, static)
)
def frozen[T: type](cls: T, /, **kwargs: Unpack[DefineOptions]) -> T: ...
@overload
@dataclass_transform(
    frozen_default=True, field_specifiers=(attrs.field, array, field, static)
)
def frozen[T: type](**kwargs: Unpack[DefineOptions]) -> Callable[[T], T]: ...
def frozen[T: type](maybe_cls: T | None = None, **kwargs) -> T | Callable:
    _warnings_hide = True
    if maybe_cls is None:
        return functools.partial(frozen, **kwargs)
    kwargs.setdefault("frozen", True)
    return define(maybe_cls, **kwargs)
