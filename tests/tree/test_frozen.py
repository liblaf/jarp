from typing import Any

import jax
import jax.tree_util as jtu
import numpy as np

import jarp

type Leaf = Any
type PyTreeDef = Any


@jarp.frozen
class A:
    x: jax.Array = jarp.array(default=0.0)
    y: str = jarp.static(default="")


def test_flatten() -> None:
    a = A()
    leaves: list[Leaf]
    treedef: PyTreeDef
    leaves, treedef = jax.tree.flatten(a)
    assert len(leaves) == 1
    np.testing.assert_allclose(leaves[0], a.x)
    a_recon: A = jax.tree.unflatten(treedef, leaves)
    np.testing.assert_allclose(a_recon.x, a.x)
    assert a_recon.y == a.y


def test_flatten_with_keys() -> None:
    a = A()
    leaves_with_path: list[tuple[jtu.KeyPath, Leaf]]
    treedef: PyTreeDef
    leaves_with_path, treedef = jax.tree.flatten_with_path(a)
    assert len(leaves_with_path) == 1
    assert leaves_with_path[0][0] == (jtu.GetAttrKey("x"),)
    leaves: list[Leaf] = [leaf for _, leaf in leaves_with_path]
    np.testing.assert_allclose(leaves[0], a.x)
    a_recon: A = jax.tree.unflatten(treedef, leaves)
    np.testing.assert_allclose(a_recon.x, a.x)
    assert a_recon.y == a.y
