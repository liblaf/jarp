import jax.numpy as jnp
import warp as wp

import jarp.warp._jax_kernel as kernel_mod
from jarp.warp import jax_kernel


def test_jax_kernel_delegates_for_concrete_kernels(monkeypatch) -> None:
    seen = {}

    def fake_jax_kernel(kernel, **kwargs):
        seen["kernel"] = kernel
        seen["kwargs"] = kwargs
        return lambda *args, **call_kwargs: (kernel.__name__, kwargs, call_kwargs)

    monkeypatch.setattr(kernel_mod.warp.jax_experimental, "jax_kernel", fake_jax_kernel)

    def kernel(x):
        del x
        return None

    wrapped = jax_kernel(kernel, launch_dims=4)
    result = wrapped(jnp.array([1]), output_dims=1)
    assert seen["kernel"] is kernel
    assert seen["kwargs"] == {"launch_dims": 4}
    assert result == ("kernel", {"launch_dims": 4}, {"output_dims": 1})


def test_jax_kernel_resolves_overloads_from_runtime_dtypes(monkeypatch) -> None:
    overload_calls = []

    def fake_overload(kernel, arg_types):
        overload_calls.append((kernel, arg_types))
        return ("overloaded", kernel, arg_types)

    def fake_jax_kernel(kernel, **kwargs):
        return lambda *args, **call_kwargs: (kernel, kwargs, call_kwargs)

    monkeypatch.setattr(kernel_mod.wp, "overload", fake_overload)
    monkeypatch.setattr(kernel_mod.warp.jax_experimental, "jax_kernel", fake_jax_kernel)

    def kernel(x):
        del x
        return None

    wrapped = jax_kernel(
        kernel,
        arg_types_factory=lambda dtype: {"x": dtype},
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
