import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import jarp


@jarp.jit
def identity[T](inputs: T) -> T:
    return inputs


@jarp.jit(filter=True)
def identity_filter[T](inputs: T) -> T:
    return inputs


@jarp.filter_jit
def identity_filter_alias[T](inputs: T) -> T:
    return inputs


@jarp.define
class A:
    @jarp.jit
    def identity[T](self, inputs: T) -> T:
        return inputs

    @jarp.jit(filter=True)
    def identity_filter[T](self, inputs: T) -> T:
        return inputs


@jarp.jit(filter=True, fallback=True)
def identity_filter_fallback[T](inputs: T) -> T:
    return inputs


@jarp.fallback_jit
def identity_filter_fallback_alias[T](inputs: T) -> T:
    return inputs


def test_jit() -> None:
    data: Array = jnp.zeros((7,))
    output_data = identity(data)
    np.testing.assert_allclose(output_data, data)


def test_jit_filter() -> None:
    data: Array = jnp.zeros((7,))
    meta: str = "meta"
    output_data, output_meta = identity_filter((data, meta))
    np.testing.assert_allclose(output_data, data)
    assert output_meta == meta


def test_filter_jit_alias() -> None:
    data: Array = jnp.zeros((7,))
    meta: str = "meta"
    output_data, output_meta = identity_filter_alias((data, meta))
    np.testing.assert_allclose(output_data, data)
    assert output_meta == meta


def test_jit_method() -> None:
    a = A()
    data: Array = jnp.zeros((7,))
    output_data = a.identity(data)
    np.testing.assert_allclose(output_data, data)

    output_data = A.identity(a, data)
    np.testing.assert_allclose(output_data, data)


def test_jit_filter_method() -> None:
    a = A()
    data: Array = jnp.zeros((7,))
    meta: str = "meta"
    output_data, output_meta = a.identity_filter((data, meta))
    np.testing.assert_allclose(output_data, data)
    assert output_meta == meta

    output_data, output_meta = A.identity_filter(a, (data, meta))
    np.testing.assert_allclose(output_data, data)
    assert output_meta == meta


def test_jit_filter_fallback_identity() -> None:
    data: Array = jnp.zeros((7,))
    meta: str = "meta"
    output_data, output_meta = identity_filter_fallback((data, meta))
    np.testing.assert_allclose(output_data, data)
    assert output_meta == meta


def test_fallback_jit_alias_identity() -> None:
    data: Array = jnp.zeros((7,))
    meta: str = "meta"
    output_data, output_meta = identity_filter_fallback_alias((data, meta))
    np.testing.assert_allclose(output_data, data)
    assert output_meta == meta


def test_jit_filter_fallback_reuses_eager_for_unsupported_metadata() -> None:
    @jarp.jit(filter=True, fallback=True)
    def branch(x: Array, mode: str) -> Array:
        if mode == "python":
            if x > 0:
                return x
            return -x
        return jnp.abs(x)

    inner = branch.inner
    inner_calls = 0

    def counted_inner(*args: object) -> object:
        nonlocal inner_calls
        inner_calls += 1
        return inner(*args)

    branch.inner = counted_inner
    data: Array = jnp.array(1)

    output_data = branch(data, "python")
    np.testing.assert_allclose(output_data, data)
    assert inner_calls == 1
    assert len(branch.unsupported) == 1

    output_data = branch(data, "python")
    np.testing.assert_allclose(output_data, data)
    assert inner_calls == 1
    assert len(branch.unsupported) == 1


def test_jit_filter_fallback_caches_by_metadata() -> None:
    @jarp.jit(filter=True, fallback=True)
    def branch(x: Array, mode: str) -> Array:
        if mode == "python":
            if x > 0:
                return x
            return -x
        return jnp.abs(x)

    inner = branch.inner
    inner_calls = 0

    def counted_inner(*args: object) -> object:
        nonlocal inner_calls
        inner_calls += 1
        return inner(*args)

    branch.inner = counted_inner
    data: Array = jnp.array(-1)

    output_data = branch(data, "python")
    np.testing.assert_allclose(output_data, jnp.array(1))
    assert inner_calls == 1
    assert len(branch.unsupported) == 1

    output_data = branch(data, "jax")
    np.testing.assert_allclose(output_data, jnp.array(1))
    assert inner_calls == 2
    assert len(branch.unsupported) == 1

    output_data = branch(data, "jax")
    np.testing.assert_allclose(output_data, jnp.array(1))
    assert inner_calls == 3
    assert len(branch.unsupported) == 1


def test_jit_filter_fallback_method() -> None:
    @jarp.define
    class B:
        @jarp.jit(filter=True, fallback=True)
        def branch(self, x: Array, mode: str) -> Array:
            if mode == "python":
                if x > 0:
                    return x
                return -x
            return jnp.abs(x)

    inner = B.branch.inner
    inner_calls = 0

    def counted_inner(*args: object) -> object:
        nonlocal inner_calls
        inner_calls += 1
        return inner(*args)

    B.branch.inner = counted_inner
    b = B()
    data: Array = jnp.array(1)

    output_data = b.branch(data, "python")
    np.testing.assert_allclose(output_data, data)
    assert inner_calls == 1

    output_data = B.branch(b, data, "python")
    np.testing.assert_allclose(output_data, data)
    assert inner_calls == 1


def test_jit_filter_fallback_does_not_swallow_non_jax_errors() -> None:
    @jarp.jit(filter=True, fallback=True)
    def bug(x: Array, mode: str) -> Array:
        if mode == "bug":
            msg = "boom"
            raise RuntimeError(msg)
        return x

    data: Array = jnp.array(1)
    with pytest.raises(RuntimeError, match="boom"):
        bug(data, "bug")


def test_jit_filter_fallback_requires_filter() -> None:
    with pytest.raises(ValueError, match="requires `filter=True`"):

        @jarp.jit(fallback=True)
        def _invalid(x: Array) -> Array:
            return x

    with pytest.raises(ValueError, match="requires `filter=True`"):

        @jarp.jit(filter=False, fallback=True)
        def _also_invalid(x: Array) -> Array:
            return x
