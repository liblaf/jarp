import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jarp


@jarp.define
class Example:
    data: object = jarp.array(default=0.0)
    meta: str = jarp.static(default="x")
    maybe: object = jarp.auto(default="y")


@jarp.frozen_static
class Config:
    name: str = "prod"


def test_define_auto_field_switches_between_data_and_metadata() -> None:
    static_leaves, _ = jax.tree.flatten(Example())
    dynamic_leaves, _ = jax.tree.flatten(Example(maybe=jnp.array([2])))

    assert len(static_leaves) == 1
    assert len(dynamic_leaves) == 2
    np.testing.assert_array_equal(dynamic_leaves[1], jnp.array([2]))


def test_frozen_static_registers_instances_as_static_leaves() -> None:
    config = Config()
    leaves, treedef = jax.tree.flatten(config)
    assert leaves == []
    assert jax.tree.unflatten(treedef, leaves) == config


def test_define_warns_for_non_frozen_static_classes() -> None:
    with pytest.warns(UserWarning, match="not frozen"):

        @jarp.define(pytree="static")
        class WarningConfig:
            name: str = "dev"
