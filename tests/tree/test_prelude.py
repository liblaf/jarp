import types
from typing import Any

import jax

import jarp

jarp.register_pytree_prelude()


@jarp.define
class A:
    def method(self) -> None: ...


def test_pytree_method() -> None:
    a = A()
    leaves: list[Any]
    treedef: Any
    leaves, treedef = jax.tree.flatten(a.method)
    assert len(leaves) == 1
    assert leaves[0] == a
    method_recon: types.MethodType = jax.tree.unflatten(treedef, leaves)
    assert method_recon.__self__ == a
    assert method_recon.__func__ == A.method

    leaves_with_path: list[tuple[Any, Any]]
    leaves_with_path, treedef = jax.tree.flatten_with_path(a.method)
    assert len(leaves_with_path) == 1
    paths: list[Any] = [path for path, _ in leaves_with_path]
    leaves: list[Any] = [leaf for _, leaf in leaves_with_path]
    assert paths[0] == jax.tree_util.GetAttrKey("__self__")
    assert leaves[0] == a
    method_recon: types.MethodType = jax.tree.unflatten(treedef, leaves)
    assert method_recon.__self__ == a
    assert method_recon.__func__ == A.method
