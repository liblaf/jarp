import jax.numpy as jnp
import warp as wp

import jarp.warp._jax_callable as callable_mod
from jarp.warp import jax_callable


def test_jax_callable_delegates_for_concrete_functions(monkeypatch) -> None:
    seen = {}

    def fake_jax_callable(func, **kwargs):
        seen["func"] = func
        seen["kwargs"] = kwargs
        return lambda *args, **call_kwargs: (
            func.__name__,
            args[0].tolist(),
            kwargs,
            call_kwargs,
        )

    monkeypatch.setattr(callable_mod.warp.jax_experimental, "jax_callable", fake_jax_callable)

    def kernel(x):
        del x
        return None

    wrapped = jax_callable(kernel, num_outputs=1)
    result = wrapped(jnp.array([1, 2]), output_dims=2)
    assert seen["func"] is kernel
    assert seen["kwargs"] == {"num_outputs": 1}
    assert result == ("kernel", [1, 2], {"num_outputs": 1}, {"output_dims": 2})


def test_jax_callable_caches_generic_factories_by_runtime_dtype(monkeypatch) -> None:
    factory_calls = []
    adapter_calls = []

    def fake_jax_callable(func, **kwargs):
        def wrapped(*args, **call_kwargs):
            adapter_calls.append((func, kwargs, call_kwargs))
            return func(*args), kwargs, call_kwargs

        return wrapped

    monkeypatch.setattr(callable_mod.warp.jax_experimental, "jax_callable", fake_jax_callable)

    def factory(dtype):
        factory_calls.append(dtype)
        return lambda *args: dtype

    wrapped = jax_callable(factory, generic=True, num_outputs=1)
    assert wrapped(jnp.array([1], dtype=jnp.int32))[0] is wp.int32
    assert wrapped(jnp.array([2], dtype=jnp.int32))[0] is wp.int32
    assert wrapped(jnp.array([1.0], dtype=jnp.float32))[0] is wp.float32
    assert factory_calls == [wp.int32, wp.float32]
    assert len(adapter_calls) == 3
