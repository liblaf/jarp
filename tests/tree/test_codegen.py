import inspect

import attrs
import jax
import jax.numpy as jnp
import numpy as np

import jarp


@attrs.define
class Example:
    data: object = attrs.field(factory=lambda: jnp.array([1]))
    label: str = "meta"
    maybe: object = "static"


def test_codegen_pytree_functions_round_trip_and_cache_source() -> None:
    funcs = jarp.tree.codegen.codegen_pytree_functions(
        Example,
        data_fields=["data"],
        meta_fields=["label"],
        auto_fields=["maybe"],
        filter_spec=lambda value: isinstance(value, jax.Array),
    )
    obj = Example(maybe=jnp.array([2]))
    rebuilt = funcs.unflatten(*reversed(funcs.flatten(obj)))

    np.testing.assert_array_equal(rebuilt.data, obj.data)
    np.testing.assert_array_equal(rebuilt.maybe, obj.maybe)
    assert funcs.flatten.__module__ == Example.__module__
    assert "def flatten" in inspect.getsource(funcs.flatten)
