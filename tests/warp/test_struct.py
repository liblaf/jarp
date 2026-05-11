from typing import Any

import jax
import warp as wp

from liblaf import jarp
from liblaf.jarp import warp


def assert_array_field_dtype(struct_type: Any, field: str, dtype: Any) -> None:
    field_type = struct_type.vars[field].type

    assert field_type.ndim == 1
    assert wp.types.types_equal(field_type.dtype, dtype)


def test_top_level_struct_export_matches_warp_submodule() -> None:
    assert jarp.struct is warp.struct


def test_struct_decorates_plain_classes_immediately() -> None:
    @warp.struct
    class PlainStruct:
        x: wp.float32

    plain_struct: Any = PlainStruct

    assert plain_struct.cls.__name__ == "PlainStruct"
    assert plain_struct.vars["x"].type is wp.float32
    assert type(plain_struct()) is plain_struct.instance_type


def test_generic_struct_specializes_and_caches_by_dtype() -> None:
    factory_calls: list[Any] = []

    @warp.struct
    class GenericParticle[T]:
        @classmethod
        def __annotations_factory__(cls, dtype: Any) -> dict[str, Any]:
            factory_calls.append(dtype)
            return {
                "position": wp.array1d(dtype=wp.types.vector(3, dtype)),
                "basis": wp.array1d(dtype=wp.types.matrix((2, 3), dtype)),
            }

    particle32 = GenericParticle[wp.float32]

    assert GenericParticle[wp.float32] is particle32
    assert factory_calls == [wp.float32]
    assert_array_field_dtype(particle32, "position", wp.types.vector(3, wp.float32))
    assert_array_field_dtype(particle32, "basis", wp.types.matrix((2, 3), wp.float32))

    particle64 = GenericParticle[wp.float64]

    assert particle64 is not particle32
    assert factory_calls == [wp.float32, wp.float64]
    assert_array_field_dtype(particle64, "position", wp.types.vector(3, wp.float64))
    assert_array_field_dtype(particle64, "basis", wp.types.matrix((2, 3), wp.float64))


def test_generic_struct_default_constructor_uses_active_jax_float_dtype() -> None:
    @warp.struct
    class DefaultParticle[T]:
        @classmethod
        def __annotations_factory__(cls, dtype: Any) -> dict[str, Any]:
            return {
                "position": wp.array1d(dtype=wp.types.vector(3, dtype)),
            }

    default_particle: Any = DefaultParticle
    original = jax.config.read("jax_enable_x64")
    disabled = False
    enabled = True
    try:
        jax.config.update("jax_enable_x64", disabled)
        assert type(default_particle()) is default_particle[wp.float32].instance_type

        jax.config.update("jax_enable_x64", enabled)
        assert type(default_particle()) is default_particle[wp.float64].instance_type
    finally:
        jax.config.update("jax_enable_x64", original)
