from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
import warp as wp

import jarp


def test_to_warp_converts_numpy_and_jax_arrays() -> None:
    vector = jarp.to_warp(np.zeros((5, 3), np.float32), (-1, Any))
    matrix = jarp.to_warp(jnp.zeros((2, 3, 3), jnp.float32), (-1, -1, Any))

    assert vector.shape == (5,)
    assert vector.dtype == wp.types.vector(3, wp.float32)
    assert matrix.shape == (2,)
    assert matrix.dtype == wp.types.matrix((3, 3), wp.float32)


def test_to_warp_reuses_compatible_warp_arrays() -> None:
    arr = wp.from_numpy(np.zeros((4,), np.float32))
    assert jarp.to_warp(arr) is arr

    with pytest.raises(ValueError, match="Cannot convert"):
        jarp.to_warp(arr, wp.float64)


def test_to_warp_can_mark_jax_arrays_as_differentiable() -> None:
    arr = jarp.to_warp(jnp.zeros((4,), jnp.float32), requires_grad=True)
    assert arr.requires_grad


def test_to_warp_rejects_unsupported_inputs() -> None:
    with pytest.raises(TypeError):
        jarp.to_warp("nope")
