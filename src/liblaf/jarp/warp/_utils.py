import jax
import warp as wp
from jax import Array

from ._types import WarpScalarDType


def dtypes_from_args(*args: Array) -> list[WarpScalarDType]:
    return [wp.dtype_from_jax(jax.dtypes.result_type(arr)) for arr in args]
