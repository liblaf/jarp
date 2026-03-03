# ref: <https://numpy.org/doc/stable/reference/arrays.scalars.html>

import re
import sys
import warnings

import jax
import warp as wp


def _floating() -> type:
    return wp.float64 if jax.config.read("jax_enable_x64") else wp.float32


def vector(length: int) -> type:
    return wp.types.vector(length, _floating())


def matrix(shape: tuple[int, int]) -> type:
    return wp.types.matrix(shape, _floating())


def __getattr__(name: str) -> type:
    if name in {"float", "float_"}:
        warnings.warn(
            f"{__name__}.{name} is deprecated, use {__name__}.floating instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return _floating()
    if name == "floating":
        return _floating()
    if (result := re.fullmatch(r"vec(?P<length>[1-9])", name)) is not None:
        length = int(result.group("length"))
        return wp.types.vector(length, _floating())
    if (result := re.fullmatch(r"mat(?P<rows>[1-9])(?P<cols>[1-9])", name)) is not None:
        rows = int(result.group("rows"))
        cols = int(result.group("cols"))
        return wp.types.matrix((rows, cols), _floating())
    msg: str = f"module '{__name__}' has no attribute '{name}'"
    raise AttributeError(msg, name=name, obj=sys.modules[__name__])
