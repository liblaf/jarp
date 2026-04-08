from typing import cast

import jax.numpy as jnp
from jax import Array

from jarp import tree


@tree.frozen_static
class Marker:
    name: str


def test_is_data_and_is_leaf_follow_the_current_partition_rules() -> None:
    assert tree.is_data(jnp.array([1]))
    assert tree.is_data(None)
    assert tree.is_data([1, 2])
    assert tree.is_data(Marker("x"))
    assert not tree.is_data("tag")

    assert tree.is_leaf(jnp.array([1]))
    assert tree.is_leaf(None)
    assert not tree.is_leaf([1, 2])
    assert not tree.is_leaf(Marker("x"))


def test_partition_and_combine_round_trip_mixed_trees() -> None:
    value = {"array": jnp.array([1, 2]), "meta": "tag", "none": None}
    data_leaves, aux = tree.partition(value)
    rebuilt = tree.combine(data_leaves, aux)
    assert cast("Array", rebuilt["array"]).tolist() == [1, 2]
    assert rebuilt["meta"] == "tag"
    assert rebuilt["none"] is None


def test_partition_leaves_separates_arrays_from_metadata() -> None:
    data_leaves, meta_leaves = tree.partition_leaves([jnp.array([1]), "tag", None])
    assert cast("Array", data_leaves[0]).tolist() == [1]
    assert data_leaves[1:] == [None, None]
    assert meta_leaves == [None, "tag", None]
