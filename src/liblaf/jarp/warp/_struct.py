"""Decorators for dtype-specialized Warp structs."""

import functools
from typing import Any, cast

import warp as wp

from . import types as wpt


def struct[T: type](cls: T) -> T:
    """Decorate a class as a Warp struct.

    Plain classes are forwarded to `warp.struct`. Classes that define
    `__annotations_factory__(dtype)` stay generic: `MyStruct[wp.float64]`
    builds and caches a specialized Warp struct from the factory annotations,
    while `MyStruct()` instantiates `MyStruct[liblaf.jarp.warp.types.floating]`
    so the default follows JAX's active precision mode.

    Args:
        cls: Class to decorate.

    Returns:
        The Warp struct for plain classes, or the original generic class with
        dtype subscription and default construction hooks installed.
    """
    if not hasattr(cls, "__annotations_factory__"):
        return cast("T", wp.struct(cls))

    @functools.cache
    def __class_getitem__(cls: T, key: Any) -> T:  # noqa: N807
        c: type = type(
            cls.__name__,
            (cls,),
            {
                "__module__": cls.__module__,
                "__qualname__": cls.__qualname__,
                "__annotations__": cls.__annotations_factory__(key),  # ty:ignore[unresolved-attribute]
            },
        )
        return cast("T", wp.struct(c, module="unique"))

    def __new__(owner: type) -> object:  # noqa: N807
        if owner is cls:
            return __class_getitem__(cls, wpt.floating)()
        return object.__new__(owner)

    cls.__class_getitem__ = classmethod(__class_getitem__)  # ty:ignore[invalid-assignment]
    cls.__new__ = staticmethod(__new__)  # ty:ignore[invalid-assignment]
    return cls
