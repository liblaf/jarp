import jax.numpy as jnp
import numpy as np
from jax import Array

import jarp


@jarp.jit
def identity[T](inputs: T) -> T:
    return inputs


@jarp.jit(filter=True)
def identity_filter[T](inputs: T) -> T:
    return inputs


@jarp.define
class A:
    @jarp.jit
    def identity[T](self, inputs: T) -> T:
        return inputs

    @jarp.jit(filter=True)
    def identity_filter[T](self, inputs: T) -> T:
        return inputs


def test_jit() -> None:
    data: Array = jnp.zeros((7,))
    output_data: Array
    output_data = identity(data)
    np.testing.assert_allclose(output_data, data)


def test_jit_filter() -> None:
    data: Array = jnp.zeros((7,))
    meta: str = "meta"
    output_data: Array
    output_meta: str
    output_data, output_meta = identity_filter((data, meta))
    np.testing.assert_allclose(output_data, data)
    assert output_meta == meta


def test_jit_method() -> None:
    a = A()
    data: Array = jnp.zeros((7,))
    output_data: Array
    output_data = a.identity(data)
    np.testing.assert_allclose(output_data, data)

    output_data = A.identity(a, data)
    np.testing.assert_allclose(output_data, data)


def test_jit_filter_method() -> None:
    a = A()
    data: Array = jnp.zeros((7,))
    meta: str = "meta"
    output_data: Array
    output_meta: str
    output_data, output_meta = a.identity_filter((data, meta))
    np.testing.assert_allclose(output_data, data)
    assert output_meta == meta

    output_data, output_meta = A.identity_filter(a, (data, meta))
    np.testing.assert_allclose(output_data, data)
    assert output_meta == meta
