import functools
import operator
from collections.abc import Callable, Sequence
from typing import Any, Literal, TypedDict

import attrs
import jax
import jax.numpy as jnp
import numpy as np
import warp as wp


class ToWarpOptions(TypedDict, total=False):
    shape: Sequence[int] | None
    device: wp.DeviceLike | None
    requires_grad: bool


def to_warp(
    arr: Any, dtype: Literal["vector", "matrix"] | Any = None, **kwargs
) -> wp.array:
    adapter: _Adapter = _get_adapter(arr)
    match dtype:
        case "vector":
            length: int = adapter.shape(arr)[-1]
            dtype = wp.types.vector(length, adapter.dtype_from(arr.dtype))
        case "matrix":
            shape: tuple[int, int] = adapter.shape(arr)[-2], adapter.shape(arr)[-1]
            dtype = wp.types.matrix(shape, adapter.dtype_from(arr.dtype))
    return adapter.array_from(arr, dtype, **kwargs)


@attrs.frozen
class _Adapter:
    array_from: Callable[..., wp.array]
    dtype_from: Callable[..., Any]
    shape: Callable[[Any], tuple[int, ...]] = operator.attrgetter("shape")


@functools.singledispatch
def _get_adapter(arr: Any) -> _Adapter:
    raise NotImplementedError


@_get_adapter.register(np.ndarray)
def _get_adapter_numpy(_arr: np.ndarray) -> _Adapter:
    return _Adapter(
        array_from=wp.from_numpy, dtype_from=wp.dtype_from_numpy, shape=np.shape
    )


@_get_adapter.register(jax.Array)
def _get_adapter_jax(_arr: jax.Array) -> _Adapter:
    return _Adapter(
        array_from=wp.from_jax, dtype_from=wp.dtype_from_jax, shape=jnp.shape
    )
