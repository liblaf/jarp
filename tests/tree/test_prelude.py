import types

import jax
import jax.tree_util as jtu
import numpy as np
import warp as wp

import jarp

jarp.register_pytree_prelude()
jarp.register_pytree_prelude()


@jarp.define
class Example:
    data: object = jarp.array(default=0.0)

    def method(self):
        return self.data


def test_register_pytree_prelude_handles_bound_methods() -> None:
    obj = Example()
    leaves_with_path, treedef = jax.tree.flatten_with_path(obj.method)
    rebuilt = jax.tree.unflatten(treedef, [leaf for _, leaf in leaves_with_path])

    assert leaves_with_path[0][0] == (
        jtu.GetAttrKey("__self__"),
        jtu.GetAttrKey("data"),
    )
    assert isinstance(rebuilt, types.MethodType)
    np.testing.assert_array_equal(rebuilt(), obj.method())


def test_register_pytree_prelude_marks_warp_arrays_static() -> None:
    arr = wp.from_numpy(np.zeros((2,), np.float32))
    leaves, treedef = jax.tree.flatten(arr)
    assert leaves == []
    assert jax.tree.unflatten(treedef, leaves) is arr
