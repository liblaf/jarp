import functools
from collections.abc import Sequence
from typing import Any

import jax
import numpy as np
import warp as wp


@functools.singledispatch
def to_warp(arr: Any, *_args, **_kwargs) -> wp.array:
    raise TypeError(arr)


def _convert_dtype(dtype: Any, arr_shape: Sequence[int], arr_dtype: Any) -> Any:
    match dtype:
        case (length, dtype):
            if length == -1:
                length: int = arr_shape[-1]
            if dtype is None or dtype is Any:
                dtype = arr_dtype
            return wp.types.vector(length, dtype)
        case (rows, cols, dtype):
            if rows == -1:
                rows: int = arr_shape[-2]
            if cols == -1:
                cols: int = arr_shape[-1]
            if dtype is None or dtype is Any:
                dtype = arr_dtype
            return wp.types.matrix((rows, cols), dtype)
        case _:
            return dtype


@to_warp.register(np.ndarray)
def _to_warp_numpy(arr: np.ndarray, dtype: Any = None, **kwargs) -> wp.array:
    dtype = _convert_dtype(dtype, arr.shape, wp.dtype_from_numpy(arr.dtype))
    return wp.from_numpy(arr, dtype, **kwargs)


@to_warp.register(jax.Array)
def _to_warp_jax(arr: jax.Array, dtype: Any = None, **kwargs) -> wp.array:
    dtype = _convert_dtype(dtype, arr.shape, wp.dtype_from_jax(arr.dtype))
    requires_grad: bool = kwargs.pop("requires_grad", False)
    arr_wp: wp.array = wp.from_jax(arr, dtype, **kwargs)
    if requires_grad:
        arr_wp.requires_grad = True
    return arr_wp
