import jax
import jax.numpy as jnp
import pytest

from jarp import tree


def test_frozen_registers_data_and_static_fields() -> None:
    @tree.frozen
    class Example:
        data: jax.Array = tree.array()
        auto_value: object = tree.auto()
        meta: str = tree.static(default="tag")

    leaves, treedef = jax.tree.flatten(Example(jnp.array([1, 2]), auto_value="meta"))
    assert len(leaves) == 1

    rebuilt = jax.tree.unflatten(treedef, leaves)
    assert rebuilt.data.tolist() == [1, 2]
    assert rebuilt.auto_value == "meta"
    assert rebuilt.meta == "tag"


def test_frozen_static_registers_instances_as_static_nodes() -> None:
    @tree.frozen_static
    class Config:
        name: str

    leaves, _ = jax.tree.flatten(Config("x"))
    assert leaves == []


def test_define_warns_for_non_frozen_static_classes() -> None:
    with pytest.warns(UserWarning, match="static class"):

        @tree.define(pytree="static")
        class Config:
            name: str

    leaves, _ = jax.tree.flatten(Config("x"))
    assert leaves == []
    assert tree.PyTreeType(True) is tree.PyTreeType.DATA
    assert tree.PyTreeType(False) is tree.PyTreeType.NONE
