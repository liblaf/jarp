import jax
import jax.numpy as jnp
import numpy as np
import warp as wp

import jarp


def test_numpy_to_warp() -> None:
    arr_np: np.ndarray = np.zeros((7,), dtype=np.float32)
    arr_wp: wp.array = jarp.to_warp(arr_np)
    assert arr_wp.shape == (7,)
    assert arr_wp.dtype == wp.float32


def test_numpy_to_warp_vector() -> None:
    arr_np: np.ndarray = np.zeros((5, 3), dtype=np.float32)
    arr_wp: wp.array = jarp.to_warp(arr_np, dtype="vector")
    assert arr_wp.shape == (5,)
    assert arr_wp.dtype == wp.types.vector(3, wp.float32)


def test_numpy_to_warp_matrix() -> None:
    arr_np: np.ndarray = np.zeros((2, 3, 3), dtype=np.float32)
    arr_wp: wp.array = jarp.to_warp(arr_np, dtype="matrix")
    assert arr_wp.shape == (2,)
    assert arr_wp.dtype == wp.types.matrix((3, 3), wp.float32)


def test_jax_to_warp() -> None:
    arr_jax: jax.Array = jnp.zeros((7,), dtype=jnp.float32)
    arr_wp: wp.array = jarp.to_warp(arr_jax)
    assert arr_wp.shape == (7,)
    assert arr_wp.dtype == wp.float32


def test_jax_to_warp_vector() -> None:
    arr_jax: jax.Array = jnp.zeros((5, 3), dtype=jnp.float32)
    arr_wp: wp.array = jarp.to_warp(arr_jax, dtype="vector")
    assert arr_wp.shape == (5,)
    assert arr_wp.dtype == wp.types.vector(3, wp.float32)


def test_jax_to_warp_matrix() -> None:
    arr_jax: jax.Array = jnp.zeros((2, 3, 3), dtype=jnp.float32)
    arr_wp: wp.array = jarp.to_warp(arr_jax, dtype="matrix")
    assert arr_wp.shape == (2,)
    assert arr_wp.dtype == wp.types.matrix((3, 3), wp.float32)
