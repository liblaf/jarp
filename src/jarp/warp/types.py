import re
import sys

import jax
import warp as wp


def _default_float_type() -> type:
    return wp.float64 if jax.config.read("jax_enable_x64") else wp.float32


def vector(length: int) -> type:
    return wp.types.vector(length, _default_float_type())


def matrix(shape: tuple[int, int]) -> type:
    return wp.types.matrix(shape, _default_float_type())


def __getattr__(name: str) -> type:
    if name == "float_":
        return _default_float_type()
    result: re.Match[str] | None = re.fullmatch(r"vec(?P<length>[1-9])", name)
    if result is not None:
        length = int(result.group("length"))
        return wp.types.vector(length, _default_float_type())
    result: re.Match[str] | None = re.fullmatch(
        r"mat(?P<rows>[1-9])(?P<cols>[1-9])", name
    )
    if result is not None:
        rows = int(result.group("rows"))
        cols = int(result.group("cols"))
        return wp.types.matrix((rows, cols), _default_float_type())
    msg: str = f"module '{__name__}' has no attribute '{name}'"
    raise AttributeError(msg, name=name, obj=sys.modules[__name__])
