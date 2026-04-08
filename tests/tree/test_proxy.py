import jax
import jax.numpy as jnp
import numpy as np

import jarp


def test_proxy_flattens_the_wrapped_value() -> None:
    proxy = jarp.PyTreeProxy((jnp.array([1]), "meta"))
    leaves, treedef = jax.tree.flatten(proxy)
    rebuilt = jax.tree.unflatten(treedef, leaves)

    assert len(leaves) == 1
    np.testing.assert_array_equal(rebuilt.__wrapped__[0], jnp.array([1]))
    assert rebuilt.__wrapped__[1] == "meta"
