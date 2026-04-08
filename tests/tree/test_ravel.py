import jax.numpy as jnp
import numpy as np

import jarp


def test_ravel_round_trips_mixed_trees() -> None:
    obj = {"a": jnp.zeros((2,)), "b": jnp.ones((3,)), "meta": "x"}
    flat, structure = jarp.ravel(obj)
    rebuilt = structure.unravel(flat)

    assert flat.shape == (5,)
    np.testing.assert_array_equal(structure.ravel(obj), flat)
    np.testing.assert_array_equal(rebuilt["a"], obj["a"])
    np.testing.assert_array_equal(rebuilt["b"], obj["b"])
    assert rebuilt["meta"] == "x"


def test_ravel_handles_leaf_structures_and_dtype_overrides() -> None:
    flat, structure = jarp.ravel("meta")
    assert structure.is_leaf
    assert flat.shape == (0,)
    assert structure.unravel(flat) == "meta"

    leaf = jnp.array([[1, 2]], dtype=jnp.int32)
    flat_leaf, leaf_structure = jarp.ravel(leaf)
    rebuilt = leaf_structure.unravel(flat_leaf, dtype=jnp.float32)

    assert rebuilt.dtype == jnp.float32
    np.testing.assert_array_equal(
        leaf_structure.ravel(jnp.array([[3, 4]], dtype=jnp.int32)),
        jnp.array([3, 4], dtype=jnp.int32),
    )
