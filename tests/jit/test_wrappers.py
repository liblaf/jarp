import logging
from typing import Any, cast

import jax.numpy as jnp
import pytest
from jax import Array

from liblaf.jarp import fallback_jit, filter_jit, tree


def test_filter_jit_partitions_callable_and_arguments() -> None:
    @filter_jit
    def affine(value: Array, *, scale: Array, label: str) -> Array:
        assert label == "ok"
        return value * scale + 1

    result = affine(jnp.array([1, 2]), scale=jnp.array([3, 4]), label="ok")

    assert result.tolist() == [4, 9]


def test_filter_jit_can_be_configured_as_decorator() -> None:
    @filter_jit(inline=True)
    def add_one(value: Array) -> Array:
        return value + 1

    assert int(add_one(jnp.array(3))) == 4


def test_filter_jit_binds_instance_methods() -> None:
    @tree.frozen
    class Scaler:
        factor: Array

        @filter_jit
        def apply(self, value: Array, *, label: str) -> Array:
            assert label == "active"
            return value * self.factor

    result = Scaler(jnp.array([2, 3])).apply(jnp.array([4, 5]), label="active")

    assert Scaler.apply is Scaler.__dict__["apply"]
    assert result.tolist() == [8, 15]


def test_fallback_jit_caches_python_fallbacks_for_failing_signatures(
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls: list[str] = []

    @fallback_jit
    def pick(index: Array, *, label: str) -> int:
        calls.append("python")
        if label == "active":
            return [10, 20][int(index)]
        return -1

    caplog.set_level(logging.ERROR, logger="liblaf.jarp._jit._fallback_jit")

    assert pick(jnp.array(1), label="active") == 20
    assert calls == ["python", "python"]

    assert pick(jnp.array(1), label="active") == 20
    assert calls == ["python", "python", "python"]
    assert len(cast("Any", pick).jit_able_cache) == 1
    assert [record.name for record in caplog.records] == [
        "liblaf.jarp._jit._fallback_jit"
    ]


def test_fallback_jit_leaves_successful_signatures_uncached() -> None:
    @fallback_jit(inline=True)
    def add_one(value: Array) -> Array:
        return value + 1

    assert int(add_one(jnp.array(3))) == 4
    assert int(add_one(jnp.array(4))) == 5
    assert cast("Any", add_one).jit_able_cache == {}


def test_fallback_jit_exposes_descriptor_on_the_class() -> None:
    @tree.frozen
    class Scaler:
        factor: Array

        @fallback_jit
        def apply(self, value: Array) -> Array:
            return value * self.factor

    assert Scaler.apply is Scaler.__dict__["apply"]
    assert Scaler(jnp.array([2, 3])).apply(jnp.array([4, 5])).tolist() == [8, 15]
