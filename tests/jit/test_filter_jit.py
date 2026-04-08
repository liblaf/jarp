import jax.numpy as jnp
import numpy as np

import jarp


@jarp.filter_jit
def identity(payload):
    return payload


@jarp.filter_jit()
def shift(x, amount):
    return x + amount


class Counter:
    @jarp.filter_jit
    def add(self, x, label):
        return x + 1, label


def test_filter_jit_preserves_static_metadata() -> None:
    data = jnp.arange(3)
    out_data, out_meta = identity((data, "tag"))
    np.testing.assert_array_equal(out_data, data)
    assert out_meta == "tag"
    assert identity.__name__ == "identity"


def test_filter_jit_supports_deferred_decoration() -> None:
    np.testing.assert_array_equal(shift(jnp.array([1, 2]), 3), jnp.array([4, 5]))


def test_filter_jit_binds_instance_methods_with_partial() -> None:
    counter = Counter()
    assert isinstance(counter.add, jarp.Partial)

    out_data, out_label = counter.add(jnp.array([1]), "x")
    np.testing.assert_array_equal(out_data, jnp.array([2]))
    assert out_label == "x"

    out_data, out_label = Counter.add(counter, jnp.array([1]), "y")
    np.testing.assert_array_equal(out_data, jnp.array([2]))
    assert out_label == "y"
