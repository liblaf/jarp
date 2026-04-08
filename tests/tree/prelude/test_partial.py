import jax
import jax.numpy as jnp

from jarp import tree


def test_partial_round_trips_through_tree_flattening() -> None:
    wrapped = tree.partial(
        lambda x, y, scale=1: (x + y) * scale,
        jnp.array([1, 2]),
        scale=3,
    )
    leaves, treedef = jax.tree.flatten(wrapped)
    assert leaves[0].tolist() == [1, 2]
    assert leaves[1] == 3

    rebuilt = jax.tree.unflatten(treedef, leaves)
    assert rebuilt(jnp.array([4, 5])).tolist() == [15, 21]
