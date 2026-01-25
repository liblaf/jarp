from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

import jarp


def test_proxy() -> None:
    data: Array = jnp.zeros(())
    meta: str = "static"
    obj: jarp.PyTreeProxy[tuple[Array, str]] = jarp.PyTreeProxy((data, meta))
    leaves: list[Any]
    treedef: Any
    leaves, treedef = jax.tree.flatten(obj)
    assert len(leaves) == 1
    np.testing.assert_allclose(leaves[0], data)
    obj_recon: jarp.PyTreeProxy[tuple[Array, str]] = jax.tree.unflatten(treedef, leaves)
    assert isinstance(obj_recon, jarp.PyTreeProxy)
    np.testing.assert_allclose(obj_recon.__wrapped__[0], data)
    assert obj_recon.__wrapped__[1] == meta
