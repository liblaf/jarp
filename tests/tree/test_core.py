from typing import Any

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import pytest
import warp as wp
from jax import Array

from liblaf.jarp import tree


def test_partition_and_combine_preserve_dynamic_and_static_parts() -> None:
    @tree.frozen
    class Model:
        weight: Array = tree.field()
        name: str = tree.static()
        bias: Array | None = tree.field(default=None)

    model = Model(weight=jnp.array([1, 2]), name="linear")

    data, aux = tree.partition(model)
    assert [leaf.tolist() if leaf is not None else None for leaf in data] == [[1, 2]]
    assert aux.meta_leaves == (None,)

    rebuilt = tree.combine([jnp.array([3, 4])], aux)
    assert isinstance(rebuilt, Model)
    assert rebuilt.weight.tolist() == [3, 4]
    assert rebuilt.name == "linear"
    assert rebuilt.bias is None


def test_is_data_includes_none_and_registered_pytrees() -> None:
    @tree.frozen
    class Box:
        value: int

    assert tree.is_leaf(None)
    assert tree.is_data(None)
    assert tree.is_data(jnp.array(1))
    assert tree.is_data([1, 2])
    assert tree.is_data(Box(1))
    assert not tree.is_data("tag")

    data_leaves, meta_leaves = tree.partition_leaves([None, "tag"])
    assert data_leaves == [None, None]
    assert meta_leaves == [None, "tag"]


def test_ravel_round_trips_mixed_tree_data() -> None:
    @tree.frozen
    class Params:
        left: Array
        label: str = tree.static()
        right: Array = tree.field()

    params = Params(left=jnp.array([1.0, 2.0]), label="p", right=jnp.array([[3.0]]))

    flat, structure = tree.ravel(params)
    assert flat.tolist() == [1.0, 2.0, 3.0]

    rebuilt = structure.unravel(jnp.array([4.0, 5.0, 6.0]))
    assert isinstance(rebuilt, Params)
    assert rebuilt.left.tolist() == [4.0, 5.0]
    assert rebuilt.right.tolist() == [[6.0]]
    assert rebuilt.label == "p"
    assert structure.ravel(rebuilt).tolist() == [4.0, 5.0, 6.0]


def test_ravel_handles_static_only_trees() -> None:
    payload = {"label": "static"}

    flat, structure = tree.ravel(payload)

    assert flat.shape == (0,)
    assert structure.unravel(flat) == payload
    assert structure.ravel(payload).shape == (0,)
    assert structure.unravel(payload) is payload


def test_ravel_rebuilds_static_leaf() -> None:
    flat, structure = tree.ravel("static")

    assert structure.is_leaf
    assert flat.shape == (0,)
    assert structure.unravel(flat) == "static"


def test_ravel_rebuilds_dynamic_array_leaf() -> None:
    flat, structure = tree.ravel(jnp.array([[1, 2], [3, 4]], dtype=jnp.float32))

    assert structure.is_leaf
    assert structure.shapes == ((2, 2),)
    assert flat.tolist() == [1.0, 2.0, 3.0, 4.0]

    rebuilt = structure.unravel(jnp.array([5, 6, 7, 8], dtype=jnp.float32))
    assert rebuilt.tolist() == [[5.0, 6.0], [7.0, 8.0]]
    assert structure.ravel(jnp.array([9, 10], dtype=jnp.float32)).tolist() == [
        9.0,
        10.0,
    ]


def test_auto_fields_move_between_metadata_and_data_by_value() -> None:
    @tree.frozen
    class Box:
        value: object = tree.auto()

    static_leaves, _ = jax.tree.flatten(Box("meta"))
    dynamic_leaves, _ = jax.tree.flatten(Box(jnp.array([1, 2])))

    assert static_leaves == []
    assert [leaf.tolist() for leaf in dynamic_leaves] == [[1, 2]]


def test_array_field_converts_concrete_defaults_to_arrays() -> None:
    @tree.define
    class Defaults:
        values: Array = tree.array(default=[1, 2, 3])

    defaults = Defaults()

    assert isinstance(defaults.values, Array)
    assert defaults.values.tolist() == [1, 2, 3]


def test_array_field_keeps_none_defaults_and_factories() -> None:
    @tree.define
    class Defaults:
        optional: object = tree.array(default=None)
        values: Array = tree.array(factory=lambda: jnp.array([1, 2]))

    defaults = Defaults()

    assert defaults.optional is None
    assert defaults.values.tolist() == [1, 2]


def test_field_and_pytree_type_coercions() -> None:
    truthy = True
    falsy = False

    assert tree.PyTreeType(truthy) is tree.PyTreeType.DATA
    assert tree.PyTreeType(None) is tree.PyTreeType.DATA
    assert tree.PyTreeType(falsy) is tree.PyTreeType.NONE
    assert tree.PyTreeType("STATIC") is tree.PyTreeType.STATIC

    assert tree.FieldType(truthy) is tree.FieldType.META
    assert tree.FieldType(falsy) is tree.FieldType.DATA
    assert tree.FieldType(None) is tree.FieldType.DATA
    assert tree.FieldType("auto") is tree.FieldType.AUTO
    assert bool(tree.FieldType.META)
    assert not bool(tree.FieldType.DATA)
    assert not bool(tree.FieldType.AUTO)


def test_define_rejects_unknown_pytree_mode() -> None:
    pytree: Any = "unknown"

    with pytest.raises(ValueError, match="unknown"):

        @tree.define(pytree=pytree)
        class Unknown:
            value: int


def test_define_can_register_static_classes() -> None:
    @tree.define(pytree="static", frozen=True)
    class Token:
        value: str

    leaves, treedef = jax.tree.flatten(Token("tag"))
    rebuilt = jax.tree.unflatten(treedef, leaves)

    assert leaves == []
    assert rebuilt == Token("tag")


def test_define_warns_for_mutable_static_classes() -> None:
    with pytest.warns(UserWarning, match="not frozen"):

        @tree.define(pytree="static")
        class Token:
            value: str


def test_frozen_decorator_factory_registers_data_classes() -> None:
    @tree.frozen()
    class Box:
        value: Array

    leaves, _ = jax.tree.flatten(Box(jnp.array([1, 2])))

    assert [leaf.tolist() for leaf in leaves] == [[1, 2]]


def test_register_generic_can_bypass_custom_setattr() -> None:
    class Custom:
        def __init__(self, value: Array, label: str) -> None:
            object.__setattr__(self, "value", value)
            object.__setattr__(self, "label", label)

        def __setattr__(self, name: str, value: object) -> None:
            msg = f"cannot set {name}"
            raise RuntimeError(msg)

    tree.register_generic(Custom, data_fields=("value",), meta_fields=("label",))

    leaves, treedef = jax.tree.flatten(Custom(jnp.array([1, 2]), "tag"))
    rebuilt = jax.tree.unflatten(treedef, leaves)

    assert [leaf.tolist() for leaf in leaves] == [[1, 2]]
    assert isinstance(rebuilt, Custom)
    assert rebuilt.value.tolist() == [1, 2]
    assert rebuilt.label == "tag"


def test_pytree_proxy_flattens_wrapped_value() -> None:
    proxy = tree.PyTreeProxy({"x": jnp.array([1, 2]), "meta": "tag"})

    leaves, treedef = jax.tree.flatten(proxy)
    rebuilt = jax.tree.unflatten(treedef, leaves)

    assert rebuilt.__wrapped__["x"].tolist() == [1, 2]
    assert rebuilt.__wrapped__["meta"] == "tag"


def test_partial_splits_wrapped_callable_from_bound_arguments() -> None:
    @tree.frozen
    class Scale:
        factor: Array

        def __call__(self, value: Array) -> Array:
            return value * self.factor

    partial = tree.partial(Scale(jnp.array([2, 3])), jnp.array([4, 5]))

    leaves, treedef = jax.tree.flatten(partial)
    assert [leaf.tolist() for leaf in leaves] == [[4, 5], [2, 3]]

    rebuilt = jax.tree.unflatten(treedef, leaves)
    assert rebuilt().tolist() == [8, 15]


def test_partial_reports_key_paths_for_bound_arguments_and_callable_data() -> None:
    @tree.frozen
    class Scale:
        factor: Array

        def __call__(self, value: Array, *, label: str) -> Array:
            assert label == "active"
            return value * self.factor

    partial = tree.partial(Scale(jnp.array([2, 3])), jnp.array([4, 5]), label="active")

    keyed_leaves, _treedef = jtu.tree_flatten_with_path(partial)

    assert keyed_leaves[0][0][0].name == "_self_args"
    assert keyed_leaves[0][0][1].idx == 0
    assert keyed_leaves[0][1].tolist() == [4, 5]
    assert keyed_leaves[1][0][0].name == "_self_kwargs"
    assert keyed_leaves[1][0][1].key == "label"
    assert keyed_leaves[1][1] == "active"
    assert keyed_leaves[2][0][0].name == "__wrapped__"
    assert keyed_leaves[2][0][1].name == "factor"
    assert keyed_leaves[2][1].tolist() == [2, 3]


def test_prelude_pytrees_are_registered_by_importing_tree() -> None:
    @tree.frozen
    class Scale:
        offset: Array

        def add(self, value: Array) -> Array:
            return value + self.offset

    method = Scale(jnp.array([1, 2])).add

    leaves, treedef = jax.tree.flatten(method)
    assert [leaf.tolist() for leaf in leaves] == [[1, 2]]

    rebuilt = jax.tree.unflatten(treedef, leaves)
    assert rebuilt(jnp.array([3, 4])).tolist() == [4, 6]
    assert jtu.is_tree_node(wp.array)


def test_bound_methods_report_key_paths_for_self_data() -> None:
    @tree.frozen
    class Scale:
        offset: Array

        def add(self, value: Array) -> Array:
            return value + self.offset

    method = Scale(jnp.array([1, 2])).add

    keyed_leaves, _treedef = jtu.tree_flatten_with_path(method)

    assert len(keyed_leaves) == 1
    assert keyed_leaves[0][0][0].name == "__self__"
    assert keyed_leaves[0][0][1].name == "offset"
    assert keyed_leaves[0][1].tolist() == [1, 2]
