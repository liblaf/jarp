import jax.numpy as jnp

from jarp import tree


def test_ravel_round_trips_mixed_trees() -> None:
    value = {"x": jnp.array([1, 2]), "meta": "tag", "none": None}
    flat, structure = tree.ravel(value)
    assert flat.tolist() == [1, 2]

    rebuilt = structure.unravel(flat)
    assert rebuilt["x"].tolist() == [1, 2]
    assert rebuilt["meta"] == "tag"
    assert rebuilt["none"] is None


def test_structure_handles_leaf_arrays_and_dtype_overrides() -> None:
    flat, structure = tree.ravel(jnp.array([[1, 2], [3, 4]], dtype=jnp.int32))
    assert structure.is_leaf
    assert flat.tolist() == [1, 2, 3, 4]

    rebuilt = structure.unravel(jnp.array([1, 2, 3, 4]), dtype=jnp.float32)
    assert rebuilt.dtype == jnp.float32
    assert rebuilt.shape == (2, 2)
    assert structure.ravel(jnp.array([[5, 6], [7, 8]])).tolist() == [5, 6, 7, 8]
