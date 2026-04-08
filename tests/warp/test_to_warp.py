from typing import Any, cast

import jax.numpy as jnp
import numpy as np
import pytest
import warp as wp

from jarp.warp import to_warp


def test_to_warp_returns_existing_warp_arrays_when_dtypes_match() -> None:
    value = wp.from_numpy(np.arange(3, dtype=np.float32), dtype=wp.float32)
    assert to_warp(value, dtype=wp.float32) is value

    with pytest.raises(ValueError, match="Cannot convert"):
        to_warp(value, dtype=wp.float64)


def test_to_warp_applies_vector_dtype_hints_to_numpy_arrays() -> None:
    value = np.arange(6, dtype=np.float32).reshape(2, 3)
    converted = to_warp(value, dtype=(-1, None))
    assert converted.shape == (2,)
    assert wp.types.types_equal(converted.dtype, wp.types.vector(3, wp.float32))


def test_to_warp_converts_jax_arrays_and_rejects_unknown_inputs() -> None:
    converted = to_warp(
        jnp.arange(4, dtype=jnp.float64).reshape(2, 2),
        dtype=wp.float32,
        requires_grad=True,
    )
    assert converted.shape == (2, 2)
    assert wp.types.types_equal(converted.dtype, wp.float32)
    assert converted.requires_grad is True

    unknown = cast("Any", object())
    with pytest.raises(TypeError):
        to_warp(unknown)
