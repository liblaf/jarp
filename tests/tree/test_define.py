from typing import Any

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jax import Array

import jarp


@jarp.define
class A:
    a: Array = jarp.array(default=0.0)
    b: str = jarp.static(default="")
    c: Array | str = jarp.auto(default="")


def test_flatten_auto_is_data() -> None:
    a = A()
    a.c = jnp.zeros(())
    leaves: list[Array]
    leaves, _treedef = jax.tree.flatten(a)
    assert len(leaves) == 2
    np.testing.assert_allclose(leaves[0], a.a)
    np.testing.assert_allclose(leaves[1], a.c)


def test_flatten_auto_is_meta() -> None:
    a = A()
    leaves: list[Array]
    leaves, _treedef = jax.tree.flatten(a)
    assert len(leaves) == 1
    np.testing.assert_allclose(leaves[0], a.a)


def test_flatten_with_keys_auto_is_data() -> None:
    a = A()
    leaves_with_path: list[tuple[jtu.KeyPath, Array]]
    leaves_with_path, _treedef = jax.tree.flatten_with_path(a)
    assert len(leaves_with_path) == 1
    paths: list[jtu.KeyPath] = [path for path, _ in leaves_with_path]
    assert paths[0] == (jtu.GetAttrKey("a"),)
    leaves: list[Array] = [leaf for _, leaf in leaves_with_path]
    np.testing.assert_allclose(leaves[0], a.a)


def test_flatten_with_keys_auto_is_meta() -> None:
    a = A()
    a.c = jnp.zeros(())
    leaves_with_path: list[tuple[jtu.KeyPath, Array]]
    leaves_with_path, _treedef = jax.tree.flatten_with_path(a)
    assert len(leaves_with_path) == 2
    paths: list[jtu.KeyPath] = [path for path, _ in leaves_with_path]
    assert paths[0] == (jtu.GetAttrKey("a"),)
    assert paths[1] == (jtu.GetAttrKey("c"),)
    leaves: list[Array] = [leaf for _, leaf in leaves_with_path]
    np.testing.assert_allclose(leaves[0], a.a)
    np.testing.assert_allclose(leaves[1], a.c)


def test_unflatten_auto_is_data() -> None:
    a = A()
    a.c = jnp.zeros(())
    leaves: list[Array]
    treedef: Any
    leaves, treedef = jax.tree.flatten(a)
    a_recon: A = jax.tree.unflatten(treedef, leaves)
    np.testing.assert_allclose(a_recon.a, a.a)
    assert a_recon.b == a.b
    assert isinstance(a_recon.c, Array)
    np.testing.assert_allclose(a_recon.c, a.c)


def test_unflatten_auto_is_meta() -> None:
    a = A()
    leaves: list[Array]
    treedef: Any
    leaves, treedef = jax.tree.flatten(a)
    a_recon: A = jax.tree.unflatten(treedef, leaves)
    np.testing.assert_allclose(a_recon.a, a.a)
    assert a_recon.b == a.b
    assert a_recon.c == a.c
