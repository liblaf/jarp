import functools
from collections.abc import Callable
from typing import dataclass_transform, overload

import attrs
from liblaf import grapes

from ._field_specifiers import field, static
from ._flatten import register_fieldz


@overload
@dataclass_transform(field_specifiers=(attrs.field, field, static))
def define[T: type](cls: T, /, **kwargs) -> T: ...
@overload
@dataclass_transform(field_specifiers=(attrs.field, field, static))
def define[T: type](**kwargs) -> Callable[[T], T]: ...
def define[T: type](maybe_cls: T | None = None, **kwargs) -> T | Callable:
    if maybe_cls is None:
        return functools.partial(define, **kwargs)
    cls: T = grapes.attrs.define(maybe_cls, **kwargs)
    register_fieldz(cls)
    return cls


@overload
@dataclass_transform(frozen_default=True, field_specifiers=(attrs.field, field, static))
def frozen[T: type](cls: T, /, **kwargs) -> T: ...
@overload
@dataclass_transform(frozen_default=True, field_specifiers=(attrs.field, field, static))
def frozen[T: type](**kwargs) -> Callable[[T], T]: ...
def frozen[T: type](maybe_cls: T | None = None, **kwargs) -> T | Callable:
    if maybe_cls is None:
        return functools.partial(frozen, **kwargs)
    cls: T = grapes.attrs.frozen(maybe_cls, **kwargs)
    register_fieldz(cls)
    return cls
