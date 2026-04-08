from typing import Any, cast

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from jarp import fallback_jit


def test_fallback_jit_records_cache_entries_by_static_metadata() -> None:
    @fallback_jit
    def add(x: Array, offset: int = 0) -> Array:
        return x + offset

    add_wrapper = cast("Any", add)
    cache = add_wrapper.jit_able_cache
    assert add(jnp.array([1, 2]), offset=1).tolist() == [2, 3]
    assert add(jnp.array([1, 2]), offset=2).tolist() == [3, 4]
    assert len(cache) == 2
    assert set(cache.values()) == {True}


def test_fallback_jit_reuses_the_python_path_after_cached_jax_errors() -> None:
    inner_calls: dict[str, int] = {"count": 0}

    @fallback_jit
    def pack(x: Array, label: str) -> dict[str, Any]:
        return {"x": x + 1, "label": label}

    class RaisingInner:
        def __init__(self, fun_meta: Any) -> None:
            self.fun_meta = fun_meta

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            del args, kwargs
            inner_calls["count"] += 1
            msg = "bad"
            raise jax.errors.JAXTypeError(msg)

    pack_wrapper = cast("Any", pack)
    pack_inner: Any = pack_wrapper.inner
    pack_wrapper.inner = RaisingInner(pack_inner.fun_meta)
    cache = pack_wrapper.jit_able_cache
    reset_cache_entry = pack_wrapper.reset_cache_entry

    assert pack_wrapper(jnp.array([1, 2]), label="tag")["x"].tolist() == [2, 3]
    assert pack_wrapper(jnp.array([5, 6]), label="tag")["x"].tolist() == [6, 7]
    assert inner_calls["count"] == 1
    assert list(cache.values()) == [False]

    reset_cache_entry(jnp.array([0]), label="tag")
    assert cache == {}

    assert pack_wrapper(jnp.array([7, 8]), label="tag")["x"].tolist() == [8, 9]
    assert inner_calls["count"] == 2


def test_fallback_jit_does_not_swallow_non_jax_errors() -> None:
    @fallback_jit
    def identity(x: Array) -> Array:
        return x

    class RaisingInner:
        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            del args, kwargs
            msg = "boom"
            raise RuntimeError(msg)

    identity_wrapper = cast("Any", identity)
    identity_wrapper.inner = RaisingInner()

    with pytest.raises(RuntimeError, match="boom"):
        identity_wrapper(jnp.array([1]))
