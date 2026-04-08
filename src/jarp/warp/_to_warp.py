import functools
from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import warp as wp
from warp._src.types import type_scalar_type


@functools.singledispatch
def to_warp(arr: Any, *_args: Any, **_kwargs: Any) -> wp.array:
    """Convert a supported array object into a [`warp.array`][warp.array].

    The dispatcher supports existing Warp arrays, NumPy arrays, and JAX arrays.
    A ``dtype`` hint may be a concrete Warp dtype or a tuple that describes a
    vector or matrix dtype inferred from the trailing dimensions of ``arr``.
    Use ``(-1, Any)`` for vector inference and ``(-1, -1, Any)`` for matrix
    inference when the element type should follow the source array.

    Args:
        arr: Array object to convert.
        *_args: Reserved for singledispatch compatibility.
        **_kwargs: Reserved for singledispatch compatibility.

    Returns:
        A Warp array view or converted array, depending on the source type.

    Raises:
        TypeError: If ``arr`` uses an unsupported type.
    """
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


@to_warp.register(wp.array)
def _to_warp_wp(arr: wp.array, dtype: Any = None, **kwargs) -> wp.array:
    del kwargs
    dtype: Any = _convert_dtype(dtype, arr.shape, arr.dtype)
    if dtype is None or wp.types.types_equal(arr.dtype, dtype):
        return arr
    msg: str = f"Cannot convert Warp array of dtype {arr.dtype} to {dtype}"
    raise ValueError(msg)


@to_warp.register(np.ndarray)
def _to_warp_numpy(arr: np.ndarray, dtype: Any = None, **kwargs) -> wp.array:
    """Convert a NumPy array into a Warp array."""
    dtype: Any = _convert_dtype(dtype, arr.shape, wp.dtype_from_numpy(arr.dtype))
    if dtype is not None:
        scalar_type: Any = type_scalar_type(dtype)
        arr = np.astype(arr, wp.dtype_to_numpy(scalar_type), copy=False)
    return wp.from_numpy(arr, dtype, **kwargs)


@to_warp.register(jax.Array)
def _to_warp_jax(arr: jax.Array, dtype: Any = None, **kwargs) -> wp.array:
    """Convert a JAX array into a Warp array."""
    dtype: Any = _convert_dtype(dtype, arr.shape, wp.dtype_from_jax(arr.dtype))
    if dtype is not None:
        scalar_type: Any = type_scalar_type(dtype)
        arr: jax.Array = jnp.astype(arr, wp.dtype_to_jax(scalar_type), copy=False)
    requires_grad: bool = kwargs.pop("requires_grad", False)
    arr_wp: wp.array = wp.from_jax(arr, dtype, **kwargs)
    if requires_grad:
        arr_wp.requires_grad = True
    return arr_wp
