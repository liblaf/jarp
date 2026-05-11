import inspect
import logging
from typing import Any, cast

import jax
import jax.numpy as jnp
import pytest

from liblaf.jarp import lax
from liblaf.jarp.lax import LaxWrapper


@pytest.mark.parametrize("name", ["cond", "fori_loop", "switch", "while_loop"])
def test_lax_wrappers_preserve_wrapped_jax_metadata(name: str) -> None:
    wrapper = getattr(lax, name)
    primitive = getattr(jax.lax, name)

    assert wrapper.__name__ == name
    assert inspect.unwrap(wrapper).__name__ == primitive.__name__
    assert inspect.signature(wrapper) == inspect.signature(primitive)


def test_cond_accepts_jax_compatible_operands() -> None:
    pred = bool(1)
    result = lax.cond(
        jnp.asarray(pred),
        lambda x: x + 1,
        lambda x: x - 1,
        jnp.array(3),
    )

    assert int(result) == 4


def test_cond_falls_back_for_python_only_branches() -> None:
    def true_fun(index: int) -> int:
        return [10, 20][index]

    def false_fun(_index: int) -> int:
        return -1

    pred_true = bool(1)
    pred_false = bool(0)

    assert lax.cond(pred_true, true_fun, false_fun, 1) == 20
    assert lax.cond(pred_false, true_fun, false_fun, 1) == -1


def test_switch_clamps_indices_for_python_fallbacks() -> None:
    branches = [
        lambda index: [10, 20][index],
        lambda index: [30, 40][index],
    ]

    assert lax.switch(-5, branches, 1) == 20
    assert lax.switch(99, branches, 1) == 40


def test_fori_loop_and_while_loop_python_fallbacks() -> None:
    assert lax.fori_loop(0, 4, lambda i, value: value + i, 0) == 6
    assert lax.while_loop(lambda value: value < 8, lambda value: value * 2, 1) == 8


def test_control_flow_helpers_fall_back_for_python_only_callbacks() -> None:
    assert lax.fori_loop(0, 3, lambda i, value: value + [10, 20, 30][i], 0) == 60
    assert lax.while_loop(
        lambda state: state[0] < 3,
        lambda state: (state[0] + 1, state[1] + [10, 20, 30][state[0]]),
        (0, 0),
    ) == (3, 60)


def test_lax_wrapper_caches_failed_signatures(caplog: pytest.LogCaptureFixture) -> None:
    calls: list[str] = []

    def wrapped(value: int) -> int:
        calls.append(f"wrapped:{value}")
        message = "boom"
        raise jax.errors.JAXTypeError(message)

    def fallback(value: int) -> int:
        calls.append("fallback")
        return value + 1

    wrapper = cast("Any", LaxWrapper(wrapped, fallback))
    caplog.set_level(logging.ERROR, logger="liblaf.jarp.lax._wrapper")

    assert wrapper(1) == 2
    assert wrapper(1) == 2
    assert calls == ["wrapped:1", "fallback", "fallback"]
    assert len(wrapper.success_cache) == 1
    assert [record.name for record in caplog.records] == ["liblaf.jarp.lax._wrapper"]


def test_lax_wrapper_retries_new_metadata_after_a_cached_failure() -> None:
    calls: list[str] = []

    def wrapped(value: int, *, mode: str) -> int:
        calls.append(f"wrapped:{mode}")
        if mode == "fallback":
            msg = "boom"
            raise jax.errors.JAXTypeError(msg)
        return value + 10

    def fallback(value: int, *, mode: str) -> int:
        calls.append(f"fallback:{mode}")
        return value + 1

    wrapper = cast("Any", LaxWrapper(wrapped, fallback))

    assert wrapper(1, mode="fallback") == 2
    assert wrapper(1, mode="fallback") == 2
    assert wrapper(1, mode="jax") == 11
    assert calls == [
        "wrapped:fallback",
        "fallback:fallback",
        "fallback:fallback",
        "wrapped:jax",
    ]
    assert len(wrapper.success_cache) == 1
