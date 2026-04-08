import jax.numpy as jnp
import warp as wp

import jarp.warp._jax_kernel as kernel_mod


def test_jax_kernel_delegates_to_warp(monkeypatch) -> None:
    calls = []

    def fake_jax_kernel(kernel, **options):
        calls.append((kernel, options))
        return lambda *args, **kwargs: (kernel, options, args, kwargs)

    monkeypatch.setattr(kernel_mod.warp.jax_experimental, "jax_kernel", fake_jax_kernel)

    def kernel(x, y):
        return x, y

    wrapped = kernel_mod.jax_kernel(kernel, num_outputs=1)
    result = wrapped(jnp.ones((2,), jnp.float32), output_dims=2)

    assert result[0] is kernel
    assert calls == [(kernel, {"num_outputs": 1})]


def test_jax_kernel_supports_decorator_usage(monkeypatch) -> None:
    monkeypatch.setattr(
        kernel_mod.warp.jax_experimental,
        "jax_kernel",
        lambda kernel, **options: (kernel, options),
    )

    def kernel(x, y):
        return x, y

    assert kernel_mod.jax_kernel(num_outputs=1)(kernel) == (kernel, {"num_outputs": 1})


def test_generic_jax_kernel_builds_overloads_from_runtime_dtype(monkeypatch) -> None:
    warp_calls = []
    overload_calls = []

    def fake_jax_kernel(kernel, **options):
        warp_calls.append((kernel, options))
        return lambda *args, **kwargs: (kernel, kwargs)

    def fake_overload(kernel, arg_types):
        overload_calls.append((kernel, arg_types))
        return ("overloaded", kernel, arg_types)

    monkeypatch.setattr(kernel_mod.warp.jax_experimental, "jax_kernel", fake_jax_kernel)
    monkeypatch.setattr(kernel_mod.wp, "overload", fake_overload)

    def kernel(x, y):
        return x, y

    wrapped = kernel_mod.jax_kernel(
        kernel,
        arg_types_factory=lambda dtype: {"x": wp.array1d[dtype]},
        num_outputs=1,
    )
    wrapped(jnp.ones((2,), jnp.float32), output_dims=3)
    wrapped(jnp.ones((2,), jnp.float64), output_dims=4)

    assert overload_calls == [
        (kernel, {"x": wp.array1d[wp.float32]}),
        (kernel, {"x": wp.array1d[wp.float64]}),
    ]
    assert len(warp_calls) == 2
