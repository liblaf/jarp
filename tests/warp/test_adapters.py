from collections.abc import Callable
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import warp as wp

from liblaf.jarp import warp
from liblaf.jarp.warp import _jax_callable as jax_callable_module
from liblaf.jarp.warp import _jax_kernel as jax_kernel_module


def test_to_warp_infers_vector_and_matrix_dtypes_from_numpy() -> None:
    vector = warp.to_warp(np.zeros((2, 3), dtype=np.float32), (-1, Any))
    matrix = warp.to_warp(np.zeros((4, 2, 3), dtype=np.float32), (-1, -1, Any))

    assert vector.shape == (2,)
    assert wp.types.types_equal(vector.dtype, wp.types.vector(3, wp.float32))
    assert matrix.shape == (4,)
    assert wp.types.types_equal(matrix.dtype, wp.types.matrix((2, 3), wp.float32))


def test_to_warp_preserves_existing_warp_arrays_and_rejects_dtype_changes() -> None:
    original = wp.from_numpy(np.zeros((2,), dtype=np.float32))

    assert warp.to_warp(original) is original
    with pytest.raises(ValueError, match="Cannot convert Warp array"):
        warp.to_warp(original, wp.float64)


def test_to_warp_sets_requires_grad_for_jax_arrays() -> None:
    wp.init()

    converted = warp.to_warp(jnp.zeros((2,), dtype=jnp.float32), requires_grad=True)

    assert converted.requires_grad
    assert converted.shape == (2,)
    assert wp.types.types_equal(converted.dtype, wp.float32)


def test_to_warp_rejects_unsupported_objects() -> None:
    value: Any = object()

    with pytest.raises(TypeError) as exc_info:
        warp.to_warp(value)

    assert exc_info.value.args == (value,)


def test_dynamic_warp_types_follow_jax_precision_setting() -> None:
    original = jax.config.read("jax_enable_x64")
    disabled = False
    enabled = True
    try:
        jax.config.update("jax_enable_x64", disabled)
        assert warp.types.floating is wp.float32
        assert wp.types.types_equal(warp.types.vec3, wp.types.vector(3, wp.float32))

        jax.config.update("jax_enable_x64", enabled)
        assert warp.types.floating is wp.float64
        assert wp.types.types_equal(
            warp.types.vector(3), wp.types.vector(3, wp.float64)
        )
        assert wp.types.types_equal(
            warp.types.matrix((2, 3)), wp.types.matrix((2, 3), wp.float64)
        )
        assert wp.types.types_equal(
            warp.types.mat23, wp.types.matrix((2, 3), wp.float64)
        )
    finally:
        jax.config.update("jax_enable_x64", original)


def test_deprecated_float_alias_warns() -> None:
    with pytest.warns(DeprecationWarning, match="floating"):
        assert warp.types.float_ is warp.types.floating


def test_unknown_dynamic_warp_type_raises_attribute_error() -> None:
    with pytest.raises(AttributeError, match="missing"):
        _ = warp.types.missing


def test_jax_callable_generic_caches_factory_by_runtime_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter_calls: list[tuple[Callable[..., object], dict[str, object]]] = []
    factory_calls: list[object] = []

    def fake_jax_callable(
        func: Callable[..., object], **options: object
    ) -> Callable[..., list[object]]:
        adapter_calls.append((func, options))

        def call(*args: object, **_kwargs: object) -> list[object]:
            return [func(*args)]

        return call

    def factory(dtype: object) -> Callable[[jax.Array], jax.Array]:
        factory_calls.append(dtype)

        def func(value: jax.Array) -> jax.Array:
            return value + 1

        return func

    monkeypatch.setattr(
        jax_callable_module.warp.jax_experimental,
        "jax_callable",
        fake_jax_callable,
    )

    wrapped = warp.jax_callable(cast("Any", factory), generic=True, num_outputs=1)

    assert int(wrapped(jnp.array(1, dtype=jnp.float32))[0]) == 2
    assert int(wrapped(jnp.array(2, dtype=jnp.float32))[0]) == 3
    assert len(factory_calls) == 1
    assert len(adapter_calls) == 2
    assert adapter_calls[0][1] == {"num_outputs": 1}


def test_jax_callable_wraps_direct_and_decorator_forms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter_calls: list[tuple[Callable[..., object], dict[str, object]]] = []

    def fake_jax_callable(
        func: Callable[..., object], **options: object
    ) -> tuple[str, Callable[..., object], dict[str, object]]:
        adapter_calls.append((func, options))
        return ("callable", func, options)

    def func() -> None:
        pass

    monkeypatch.setattr(
        jax_callable_module.warp.jax_experimental,
        "jax_callable",
        fake_jax_callable,
    )

    direct = warp.jax_callable(func, num_outputs=1)
    decorated = warp.jax_callable(num_outputs=2)(func)

    assert direct == ("callable", func, {"num_outputs": 1})
    assert decorated == ("callable", func, {"num_outputs": 2})
    assert adapter_calls == [
        (func, {"num_outputs": 1}),
        (func, {"num_outputs": 2}),
    ]


def test_jax_kernel_generic_resolves_overload_from_runtime_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    overload_calls: list[tuple[object, object]] = []
    adapter_calls: list[tuple[object, dict[str, object]]] = []

    def kernel() -> None:
        pass

    def fake_overload(kernel_obj: object, arg_types: object) -> object:
        overload_calls.append((kernel_obj, arg_types))
        return ("overload", kernel_obj, arg_types)

    def fake_jax_kernel(
        kernel_obj: object, **options: object
    ) -> Callable[..., list[object]]:
        adapter_calls.append((kernel_obj, options))

        def call(*args: object, **_kwargs: object) -> list[object]:
            return [args[0]]

        return call

    monkeypatch.setattr(jax_kernel_module.wp, "overload", fake_overload)
    monkeypatch.setattr(
        jax_kernel_module.warp.jax_experimental,
        "jax_kernel",
        fake_jax_kernel,
    )

    wrapped = warp.jax_kernel(
        kernel,
        arg_types_factory=lambda dtype: [dtype],
        num_outputs=1,
    )

    result = wrapped(jnp.array(1, dtype=jnp.float32))

    assert int(result[0]) == 1
    assert len(overload_calls) == 1
    assert overload_calls[0][0] is kernel
    assert adapter_calls == [
        (("overload", overload_calls[0][0], overload_calls[0][1]), {"num_outputs": 1})
    ]


def test_jax_kernel_wraps_direct_and_decorator_forms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter_calls: list[tuple[Callable[..., object], dict[str, object]]] = []

    def fake_jax_kernel(
        func: Callable[..., object], **options: object
    ) -> tuple[str, Callable[..., object], dict[str, object]]:
        adapter_calls.append((func, options))
        return ("kernel", func, options)

    def kernel() -> None:
        pass

    monkeypatch.setattr(
        jax_kernel_module.warp.jax_experimental,
        "jax_kernel",
        fake_jax_kernel,
    )

    direct = warp.jax_kernel(kernel, num_outputs=1)
    decorated = warp.jax_kernel(num_outputs=2)(kernel)

    assert direct == ("kernel", kernel, {"num_outputs": 1})
    assert decorated == ("kernel", kernel, {"num_outputs": 2})
    assert adapter_calls == [
        (kernel, {"num_outputs": 1}),
        (kernel, {"num_outputs": 2}),
    ]
