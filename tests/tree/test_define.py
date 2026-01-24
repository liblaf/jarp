from typing import Any

import jax
import jax.tree_util as jtu

from jarp import tree

type Leaf = Any
type PyTreeDef = Any


@tree.define
class A:
    x: int
    y: str = tree.static()


def test_flatten() -> None:
    a = A(x=0, y="")
    leaves: list[Leaf]
    treedef: PyTreeDef
    leaves, treedef = jax.tree.flatten(a)
    assert leaves == [a.x]
    assert a == jax.tree.unflatten(treedef, leaves)


def test_flatten_with_keys() -> None:
    a = A(x=0, y="")
    leaves_with_path: list[tuple[jtu.KeyPath, Leaf]]
    treedef: PyTreeDef
    leaves_with_path, treedef = jax.tree.flatten_with_path(a)
    assert leaves_with_path == [((jtu.GetAttrKey("x"),), a.x)]
    leaves: list[Leaf] = [leaf for _, leaf in leaves_with_path]
    assert a == jax.tree.unflatten(treedef, leaves)
