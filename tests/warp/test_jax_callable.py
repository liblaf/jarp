import jax.numpy as jnp
import warp as wp

import jarp.warp._jax_callable as callable_mod


def test_jax_callable_delegates_to_warp(monkeypatch) -> None:
    calls = []

    def fake_jax_callable(func, **options):
        calls.append((func, options))
        return lambda *args, **kwargs: (func, options, args, kwargs)

    monkeypatch.setattr(callable_mod.warp.jax_experimental, "jax_callable", fake_jax_callable)

    def kernel(x):
        return x

    wrapped = callable_mod.jax_callable(kernel, num_outputs=2)
    result = wrapped(jnp.ones((2,), jnp.float32), output_dims=3)

    assert result[0] is kernel
    assert calls == [(kernel, {"num_outputs": 2})]


def test_jax_callable_supports_decorator_usage(monkeypatch) -> None:
    monkeypatch.setattr(
        callable_mod.warp.jax_experimental,
        "jax_callable",
        lambda func, **options: (func, options),
    )

    def kernel(x):
        return x

    assert callable_mod.jax_callable(num_outputs=1)(kernel) == (
        kernel,
        {"num_outputs": 1},
    )


def test_generic_jax_callable_caches_factories_by_runtime_dtype(monkeypatch) -> None:
    warp_calls = []
    factory_calls = []

    def fake_jax_callable(func, **options):
        warp_calls.append((func, options))
        return lambda *args, **kwargs: (func, kwargs)

    monkeypatch.setattr(callable_mod.warp.jax_experimental, "jax_callable", fake_jax_callable)

    def factory(dtype):
        factory_calls.append(dtype)

        def impl(x):
            return x

        return impl

    wrapped = callable_mod.jax_callable(factory, generic=True, num_outputs=1)
    wrapped(jnp.ones((2,), jnp.float32), output_dims=5)
    wrapped(jnp.ones((2,), jnp.float64), output_dims=6)
    wrapped(jnp.ones((3,), jnp.float32), output_dims=7)

    assert factory_calls == [wp.float32, wp.float64]
    assert len(warp_calls) == 3
