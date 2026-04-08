from collections.abc import Callable
from typing import Any, cast

import jax.numpy as jnp
import pytest
import warp as wp

import jarp.warp._jax_callable as callable_mod
from jarp.warp import jax_callable


def test_jax_callable_delegates_for_concrete_functions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def fake_jax_callable(
        func: Callable[..., Any], **kwargs: Any
    ) -> Callable[..., tuple[str, list[int], dict[str, Any], dict[str, Any]]]:
        seen["func"] = func
        seen["kwargs"] = kwargs

        def wrapped(
            *args: Any, **call_kwargs: Any
        ) -> tuple[str, list[int], dict[str, Any], dict[str, Any]]:
            func_name = cast("Any", func).__name__
            return (
                func_name,
                args[0].tolist(),
                kwargs,
                call_kwargs,
            )

        return wrapped

    monkeypatch.setattr(
        callable_mod.warp.jax_experimental,
        "jax_callable",
        fake_jax_callable,
    )

    def kernel(x: Any) -> None:
        del x

    wrapped = jax_callable(kernel, num_outputs=1)
    result = wrapped(jnp.array([1, 2]), output_dims=2)
    assert seen["func"] is kernel
    assert seen["kwargs"] == {"num_outputs": 1}
    assert result == ("kernel", [1, 2], {"num_outputs": 1}, {"output_dims": 2})


def test_jax_callable_caches_generic_factories_by_runtime_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory_calls: list[Any] = []
    adapter_calls: list[tuple[Callable[..., Any], dict[str, Any], dict[str, Any]]] = []

    def fake_jax_callable(
        func: Callable[..., Any], **kwargs: Any
    ) -> Callable[..., tuple[Any, dict[str, Any], dict[str, Any]]]:
        def wrapped(
            *args: Any, **call_kwargs: Any
        ) -> tuple[Any, dict[str, Any], dict[str, Any]]:
            adapter_calls.append((func, kwargs, call_kwargs))
            return func(*args), kwargs, call_kwargs

        return wrapped

    monkeypatch.setattr(
        callable_mod.warp.jax_experimental,
        "jax_callable",
        fake_jax_callable,
    )

    def factory(dtype: Any) -> Callable[..., Any]:
        factory_calls.append(dtype)

        def implementation(*_args: Any) -> Any:
            return dtype

        return implementation

    wrapped = jax_callable(factory, generic=True, num_outputs=1)
    assert wrapped(jnp.array([1], dtype=jnp.int32))[0] is wp.int32
    assert wrapped(jnp.array([2], dtype=jnp.int32))[0] is wp.int32
    assert wrapped(jnp.array([1.0], dtype=jnp.float32))[0] is wp.float32
    assert factory_calls == [wp.int32, wp.float32]
    assert len(adapter_calls) == 3
