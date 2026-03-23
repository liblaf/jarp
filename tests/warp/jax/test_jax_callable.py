from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
import warp as wp
from jax import Array
from jaxtyping import DTypeLike

import jarp

pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA not available")

floating = Any


@wp.kernel
def identity_kernel(x: wp.array1d[floating], y: wp.array1d[floating]) -> None:
    tid = wp.tid()
    y[tid] = x[tid]


@jarp.warp.jax_callable
def identity_callable(x: wp.array1d[wp.float32], y: wp.array1d[wp.float32]) -> None:
    wp.launch(identity_kernel, x.shape, [x], [y])


@jarp.warp.jax_callable(generic=True)
def identity_callable_generic(dtype: Any) -> Callable[..., None]:
    def identity_callable(x: wp.array1d[dtype], y: wp.array1d[dtype]) -> None:
        wp.launch(identity_kernel, x.shape, [x], [y])

    return identity_callable


def test_jax_callable() -> None:
    x: Array = jnp.ones((7,), jnp.float32)
    (y,) = identity_callable(x)
    np.testing.assert_allclose(x, y)
    assert y.dtype == x.dtype


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.float32, jnp.float64])
def test_jax_callable_generic(dtype: DTypeLike) -> None:
    x: Array = jnp.ones((7,), dtype)
    (y,) = identity_callable_generic(x)
    np.testing.assert_allclose(x, y)
    assert y.dtype == x.dtype
