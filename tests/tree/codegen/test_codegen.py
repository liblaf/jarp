import jax
import jax.numpy as jnp

from jarp import tree


class Plain:
    def __init__(self, data, meta, auto):
        self.data = data
        self.meta = meta
        self.auto = auto


class Frozenish:
    def __init__(self, data, meta):
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "meta", meta)

    def __setattr__(self, name, value):
        del name, value
        raise RuntimeError("blocked")


tree.register_generic(Frozenish, data_fields=("data",), meta_fields=("meta",))


def test_codegen_pytree_functions_round_trip_without_registration() -> None:
    functions = tree.codegen.codegen_pytree_functions(
        Plain,
        data_fields=("data",),
        meta_fields=("meta",),
        auto_fields=("auto",),
    )
    value = Plain(jnp.array([1, 2]), "tag", "auto-meta")
    children, aux = functions.flatten(value)
    assert children[0].tolist() == [1, 2]
    assert children[1] is None

    children_with_keys, _ = functions.flatten_with_keys(value)
    assert [key.name for key, _child in children_with_keys] == ["data", "auto"]

    rebuilt = functions.unflatten(aux, children)
    assert rebuilt.meta == "tag"
    assert rebuilt.auto == "auto-meta"


def test_register_generic_uses_object_setattr_for_nonstandard_classes() -> None:
    value = Frozenish(jnp.array([1, 2]), "tag")
    leaves, treedef = jax.tree.flatten(value)
    rebuilt = jax.tree.unflatten(treedef, leaves)
    assert rebuilt.data.tolist() == [1, 2]
    assert rebuilt.meta == "tag"
