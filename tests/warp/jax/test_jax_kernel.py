from typing import Any, no_type_check

import jax.numpy as jnp
import numpy as np
import pytest
import warp as wp
from jax import Array
from jaxtyping import DTypeLike

import jarp

float_ = Any


@jarp.warp.jax_kernel
@wp.kernel
@no_type_check
def identity_kernel(
    x: wp.array1d(dtype=wp.float32), y: wp.array1d(dtype=wp.float32)
) -> None:
    tid = wp.tid()
    y[tid] = x[tid]


def test_jax_kernel() -> None:
    x: Array = jnp.ones((7,), jnp.float32)
    y: Array
    (y,) = identity_kernel(x)
    np.testing.assert_allclose(x, y)
    assert y.dtype == x.dtype


@jarp.warp.jax_kernel(
    arg_types_factory=lambda x_dtype: {
        "x": wp.array1d(dtype=x_dtype),
        "y": wp.array1d(dtype=x_dtype),
    }
)
@wp.kernel
@no_type_check
def identity_kernel_generic(
    x: wp.array1d(dtype=float_), y: wp.array1d(dtype=float_)
) -> None:
    tid = wp.tid()
    y[tid] = x[tid]


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.float32, jnp.float64])
def test_jax_kernel_generic(dtype: DTypeLike) -> None:
    x: Array = jnp.ones((7,), dtype)
    y: Array
    (y,) = identity_kernel_generic(x)
    np.testing.assert_allclose(x, y)
    assert y.dtype == x.dtype
