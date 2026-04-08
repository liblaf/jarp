import jax.numpy as jnp
import numpy as np

import jarp


@jarp.define
class Node:
    value: object = jarp.array(default=0.0)


def test_partition_and_combine_round_trip_mixed_trees() -> None:
    obj = (jnp.array([1, 2]), "meta", None)
    data_leaves, aux = jarp.tree.partition(obj)
    combined = jarp.tree.combine_leaves(data_leaves, aux.meta_leaves)
    rebuilt = jarp.tree.combine(data_leaves, aux)

    assert len(combined) == 2
    assert combined[1] == "meta"
    np.testing.assert_array_equal(rebuilt[0], obj[0])
    assert rebuilt[1:] == ("meta", None)


def test_is_data_recognizes_registered_pytrees() -> None:
    node = Node(jnp.array([1]))
    assert jarp.tree.is_data(node)
    assert not jarp.tree.is_leaf(node)
    assert jarp.tree.is_leaf(None)
    assert not jarp.tree.is_data("meta")
