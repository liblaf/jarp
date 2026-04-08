from collections.abc import Callable
from typing import Any, cast

import jax.numpy as jnp
import pytest
import warp as wp

import jarp.warp._jax_kernel as kernel_mod
from jarp.warp import jax_kernel


def test_jax_kernel_delegates_for_concrete_kernels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def fake_jax_kernel(
        kernel: Callable[..., Any], **kwargs: Any
    ) -> Callable[..., tuple[str, dict[str, Any], dict[str, Any]]]:
        seen["kernel"] = kernel
        seen["kwargs"] = kwargs

        def wrapped(
            *_args: Any, **call_kwargs: Any
        ) -> tuple[str, dict[str, Any], dict[str, Any]]:
            kernel_name = cast("Any", kernel).__name__
            return kernel_name, kwargs, call_kwargs

        return wrapped

    monkeypatch.setattr(
        kernel_mod.warp.jax_experimental,
        "jax_kernel",
        fake_jax_kernel,
    )

    def kernel(x: Any) -> None:
        del x

    wrapped = jax_kernel(kernel, launch_dims=4)
    result = wrapped(jnp.array([1]), output_dims=1)
    assert seen["kernel"] is kernel
    assert seen["kwargs"] == {"launch_dims": 4}
    assert result == ("kernel", {"launch_dims": 4}, {"output_dims": 1})


def test_jax_kernel_resolves_overloads_from_runtime_dtypes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    overload_calls: list[tuple[Callable[..., Any], dict[str, Any]]] = []

    def fake_overload(
        kernel: Callable[..., Any], arg_types: dict[str, Any]
    ) -> tuple[str, Callable[..., Any], dict[str, Any]]:
        overload_calls.append((kernel, arg_types))
        return ("overloaded", kernel, arg_types)

    def fake_jax_kernel(
        kernel: Callable[..., Any], **kwargs: Any
    ) -> Callable[..., tuple[Any, dict[str, Any], dict[str, Any]]]:
        def wrapped(
            *_args: Any, **call_kwargs: Any
        ) -> tuple[Any, dict[str, Any], dict[str, Any]]:
            return kernel, kwargs, call_kwargs

        return wrapped

    monkeypatch.setattr(kernel_mod.wp, "overload", fake_overload)
    monkeypatch.setattr(
        kernel_mod.warp.jax_experimental,
        "jax_kernel",
        fake_jax_kernel,
    )

    def kernel(x: Any) -> None:
        del x

    def arg_types_factory(dtype: Any) -> dict[str, Any]:
        return {"x": dtype}

    wrapped = jax_kernel(
        kernel,
        arg_types_factory=arg_types_factory,
        enable_backward=True,
    )
    overloaded_kernel, kwargs, call_kwargs = wrapped(
        jnp.array([1], dtype=jnp.int32),
        launch_dims=2,
    )
    assert overloaded_kernel[0] == "overloaded"
    assert overload_calls == [(kernel, {"x": wp.int32})]
    assert kwargs == {"enable_backward": True}
    assert call_kwargs == {"launch_dims": 2}
