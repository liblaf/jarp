import functools
from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import warp as wp
from warp._src.types import type_scalar_type


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
    dtype: Any = _convert_dtype(dtype, arr.shape, wp.dtype_from_numpy(arr.dtype))
    if dtype is not None:
        scalar_type: Any = type_scalar_type(dtype)
        arr = np.astype(arr, wp.dtype_to_numpy(scalar_type), copy=False)
    return wp.from_numpy(arr, dtype, **kwargs)


@to_warp.register(jax.Array)
def _to_warp_jax(arr: jax.Array, dtype: Any = None, **kwargs) -> wp.array:
    dtype: Any = _convert_dtype(dtype, arr.shape, wp.dtype_from_jax(arr.dtype))
    if dtype is not None:
        scalar_type: Any = type_scalar_type(dtype)
        arr: jax.Array = jnp.astype(arr, wp.dtype_to_jax(scalar_type), copy=False)
    requires_grad: bool = kwargs.pop("requires_grad", False)
    arr_wp: wp.array = wp.from_jax(arr, dtype, **kwargs)
    if requires_grad:
        arr_wp.requires_grad = True
    return arr_wp
