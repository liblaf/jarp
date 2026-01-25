from typing import Any

import attrs
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

import jarp


@attrs.define
class A:
    a: Array = attrs.field(factory=lambda: jnp.zeros(()))
    b: str = "static"
    c: Array = attrs.field(factory=lambda: jnp.ones(()))
    d: str = "static"


jarp.tree.register_generic(A, ["a"], ["b"], ["c", "d"])


def test_register_generic() -> None:
    a = A()
    leaves: list[Any]
    treedef: Any
    leaves, treedef = jax.tree.flatten(a)
    assert len(leaves) == 2
    np.testing.assert_allclose(leaves[0], a.a)
    np.testing.assert_allclose(leaves[1], a.c)

    a_recon: A = jax.tree.unflatten(treedef, leaves)
    np.testing.assert_allclose(a_recon.a, a.a)
    assert a_recon.b == a.b
    np.testing.assert_allclose(a_recon.c, a.c)
    assert a_recon.d == a.d

    leaves_with_path: list[tuple[Any, Any]]
    leaves_with_path, treedef = jax.tree.flatten_with_path(a)
    leaves: list[Any] = [leaf for _, leaf in leaves_with_path]
    assert len(leaves) == 2
    assert leaves_with_path[0][0] == (jax.tree_util.GetAttrKey("a"),)
    assert leaves_with_path[1][0] == (jax.tree_util.GetAttrKey("c"),)
    np.testing.assert_allclose(leaves[0], a.a)
    np.testing.assert_allclose(leaves[1], a.c)
    a_recon = jax.tree.unflatten(treedef, leaves)
    np.testing.assert_allclose(a_recon.a, a.a)
    assert a_recon.b == a.b
    np.testing.assert_allclose(a_recon.c, a.c)
    assert a_recon.d == a.d
