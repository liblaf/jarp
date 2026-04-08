import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

import jarp


def add(x, y, *, scale=1):
    return (x + y) * scale


@jarp.define
class Affine:
    bias: object = jarp.array(default=0.0)

    def __call__(self, x):
        return x + self.bias


def test_partial_round_trips_plain_functions() -> None:
    wrapped = jarp.partial(add, jnp.array([1]), scale=2)
    leaves, treedef = jax.tree.flatten(wrapped)
    rebuilt = jax.tree.unflatten(treedef, leaves)

    assert len(leaves) == 2
    np.testing.assert_array_equal(rebuilt(jnp.array([3])), jnp.array([8]))


def test_partial_keeps_registered_callables_in_the_tree() -> None:
    wrapped = jarp.partial(Affine(jnp.array([2])), jnp.array([3]))
    leaves_with_path, treedef = jax.tree.flatten_with_path(wrapped)
    paths = [path for path, _ in leaves_with_path]
    rebuilt = jax.tree.unflatten(treedef, [leaf for _, leaf in leaves_with_path])

    assert paths[1] == (jtu.GetAttrKey("__wrapped__"), jtu.GetAttrKey("bias"))
    np.testing.assert_array_equal(rebuilt(), jnp.array([5]))
