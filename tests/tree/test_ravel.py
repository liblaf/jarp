from typing import Any

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

import jarp


def test_ravel() -> None:
    obj: dict[str, Any] = {"a": jnp.zeros((3,)), "b": jnp.ones((4,)), "static": "foo"}
    flat: Float[Array, " N"]
    structure: jarp.Structure[dict[str, Any]]
    flat, structure = jarp.ravel(obj)
    assert flat.shape == (7,)

    # ravel flat array is identity
    np.testing.assert_allclose(flat, structure.ravel(flat))

    recon: dict[str, Any] = structure.unravel(flat)
    np.testing.assert_allclose(recon["a"], jnp.zeros((3,)))
    np.testing.assert_allclose(recon["b"], jnp.ones((4,)))
    assert recon["static"] == obj["static"]

    # unravel pytree is identity
    recon: dict[str, Any] = structure.unravel(obj)
    np.testing.assert_allclose(recon["a"], jnp.zeros((3,)))
    np.testing.assert_allclose(recon["b"], jnp.ones((4,)))
    assert recon["static"] == obj["static"]
