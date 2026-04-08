import attrs
import jax
import jax.numpy as jnp

from jarp import tree


@attrs.define
class Example:
    data: jax.Array = tree.array()
    auto_value: object = tree.auto()
    meta: str = tree.static(default="tag")


tree.register_fieldz(Example)


def test_register_fieldz_uses_field_metadata_for_auto_fields() -> None:
    dynamic = Example(jnp.array([1, 2]), auto_value=jnp.array([3, 4]))
    static = Example(jnp.array([1, 2]), auto_value="name")

    dynamic_leaves, _ = jax.tree.flatten(dynamic)
    static_leaves, _ = jax.tree.flatten(static)

    assert [leaf.tolist() for leaf in dynamic_leaves] == [[1, 2], [3, 4]]
    assert [leaf.tolist() for leaf in static_leaves] == [[1, 2]]
