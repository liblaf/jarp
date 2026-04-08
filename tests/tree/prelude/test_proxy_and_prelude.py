import jax
import jax.numpy as jnp
from jax import Array

from jarp import tree


def test_pytree_proxy_flattens_the_wrapped_value() -> None:
    proxy = tree.PyTreeProxy({"x": jnp.array([1, 2]), "meta": "tag"})
    leaves, treedef = jax.tree.flatten(proxy)
    rebuilt = jax.tree.unflatten(treedef, leaves)
    assert rebuilt.__wrapped__["x"].tolist() == [1, 2]
    assert rebuilt.__wrapped__["meta"] == "tag"


def test_register_pytree_prelude_is_idempotent_for_bound_methods() -> None:
    tree.register_pytree_prelude()
    tree.register_pytree_prelude()

    class Scale:
        def __init__(self, offset: Array) -> None:
            self.offset = offset

        def add(self, x: Array) -> Array:
            return x + self.offset

    method = Scale(jnp.array([1, 2])).add
    leaves, treedef = jax.tree.flatten(method)
    rebuilt = jax.tree.unflatten(treedef, leaves)
    assert rebuilt(jnp.array([3, 4])).tolist() == [4, 6]
