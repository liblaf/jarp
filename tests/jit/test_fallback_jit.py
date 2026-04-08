import jax
import jax.numpy as jnp
import pytest

from jarp import fallback_jit


def test_fallback_jit_records_cache_entries_by_static_metadata() -> None:
    @fallback_jit
    def add(x, offset=0):
        return x + offset

    assert add(jnp.array([1, 2]), offset=1).tolist() == [2, 3]
    assert add(jnp.array([1, 2]), offset=2).tolist() == [3, 4]
    assert len(add.jit_able_cache) == 2
    assert set(add.jit_able_cache.values()) == {True}


def test_fallback_jit_reuses_the_python_path_after_cached_jax_errors() -> None:
    inner_calls = {"count": 0}

    @fallback_jit
    def pack(x, label):
        return {"x": x + 1, "label": label}

    class RaisingInner:
        def __init__(self, fun_meta) -> None:
            self.fun_meta = fun_meta

        def __call__(self, *args, **kwargs):
            del args, kwargs
            inner_calls["count"] += 1
            raise jax.errors.JAXTypeError("bad")

    pack.inner = RaisingInner(pack.inner.fun_meta)

    assert pack(jnp.array([1, 2]), label="tag")["x"].tolist() == [2, 3]
    assert pack(jnp.array([5, 6]), label="tag")["x"].tolist() == [6, 7]
    assert inner_calls["count"] == 1
    assert list(pack.jit_able_cache.values()) == [False]

    pack.reset_cache_entry(jnp.array([0]), label="tag")
    assert pack.jit_able_cache == {}

    assert pack(jnp.array([7, 8]), label="tag")["x"].tolist() == [8, 9]
    assert inner_calls["count"] == 2


def test_fallback_jit_does_not_swallow_non_jax_errors() -> None:
    @fallback_jit
    def identity(x):
        return x

    class RaisingInner:
        def __call__(self, *args, **kwargs):
            del args, kwargs
            raise RuntimeError("boom")

    identity.inner = RaisingInner()

    with pytest.raises(RuntimeError, match="boom"):
        identity(jnp.array([1]))
