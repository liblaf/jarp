import attrs
import jax
import jax.numpy as jnp
import numpy as np

import jarp


@attrs.define
class Explicit:
    data: object = attrs.field(factory=lambda: jnp.array([1]))
    label: str = "meta"


@attrs.define
class Generic:
    data: object = attrs.field(factory=lambda: jnp.array([1]))
    maybe: object = "meta"


jarp.tree.register_fieldz(Explicit, data_fields=["data"], meta_fields=["label"])
jarp.tree.register_generic(
    Generic,
    ["data"],
    [],
    ["maybe"],
    filter_spec=lambda value: isinstance(value, jax.Array),
)


def test_register_fieldz_supports_explicit_field_groups() -> None:
    obj = Explicit()
    leaves, treedef = jax.tree.flatten(obj)
    rebuilt = jax.tree.unflatten(treedef, leaves)

    np.testing.assert_array_equal(rebuilt.data, obj.data)
    assert rebuilt.label == obj.label


def test_register_generic_handles_auto_fields() -> None:
    obj = Generic(maybe=jnp.array([2]))
    leaves, treedef = jax.tree.flatten(obj)
    rebuilt = jax.tree.unflatten(treedef, leaves)

    assert len(leaves) == 2
    np.testing.assert_array_equal(rebuilt.data, obj.data)
    np.testing.assert_array_equal(rebuilt.maybe, obj.maybe)
