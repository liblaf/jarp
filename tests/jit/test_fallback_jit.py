import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jarp


class FlakyInner:
    def __init__(self, fun_meta) -> None:
        self.fun_meta = fun_meta
        self.calls = 0

    def __call__(self, *args, **kwargs):
        del args, kwargs
        self.calls += 1
        raise jax.errors.JAXTypeError("boom")


def test_fallback_jit_supports_deferred_decoration_and_successful_calls() -> None:
    @jarp.fallback_jit()
    def add_one(x):
        return x + 1

    np.testing.assert_array_equal(add_one(jnp.array([1])), jnp.array([2]))
    assert list(add_one.jit_able_cache.values()) == [True]


def test_fallback_jit_reuses_python_path_after_cached_error(monkeypatch) -> None:
    @jarp.fallback_jit
    def add(x, mode):
        return x + (1 if mode == "one" else 2)

    inner = FlakyInner(add.inner.fun_meta)
    monkeypatch.setattr(add, "inner", inner)

    np.testing.assert_array_equal(add(jnp.array([1]), "one"), jnp.array([2]))
    np.testing.assert_array_equal(add(jnp.array([1]), "one"), jnp.array([2]))
    assert inner.calls == 1

    add.reset_cache_entry(jnp.array([1]), "one")
    np.testing.assert_array_equal(add(jnp.array([1]), "one"), jnp.array([2]))
    assert inner.calls == 2


def test_fallback_jit_caches_results_by_metadata(monkeypatch) -> None:
    @jarp.fallback_jit
    def add(x, mode):
        return x + (1 if mode == "one" else 2)

    inner = FlakyInner(add.inner.fun_meta)
    monkeypatch.setattr(add, "inner", inner)

    add(jnp.array([1]), "one")
    add(jnp.array([1]), "two")
    assert len(add.jit_able_cache) == 2


def test_fallback_jit_does_not_swallow_non_jax_errors(monkeypatch) -> None:
    @jarp.fallback_jit
    def passthrough(x):
        return x

    class BrokenInner:
        fun_meta = passthrough.inner.fun_meta

        def __call__(self, *args, **kwargs):
            del args, kwargs
            raise RuntimeError("boom")

    monkeypatch.setattr(passthrough, "inner", BrokenInner())

    with pytest.raises(RuntimeError, match="boom"):
        passthrough(jnp.array([1]))
