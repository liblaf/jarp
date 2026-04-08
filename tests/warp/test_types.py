import jax
import pytest
import warp as wp

import jarp.warp.types as types


def test_types_follow_the_active_jax_precision_mode() -> None:
    with jax.enable_x64(True):  # noqa: FBT003
        assert types.floating is wp.float64
        assert types.vec3 is wp.types.vector(3, wp.float64)
        assert types.mat22 is wp.types.matrix((2, 2), wp.float64)

    with jax.enable_x64(False):  # noqa: FBT003
        assert types.floating is wp.float32
        assert types.vector(5) is wp.types.vector(5, wp.float32)
        assert types.matrix((2, 3)) is wp.types.matrix((2, 3), wp.float32)


def test_types_warn_on_deprecated_aliases_and_unknown_names() -> None:
    with pytest.warns(DeprecationWarning):
        assert getattr(types, "float_") is types.floating

    with pytest.raises(AttributeError, match="has no attribute"):
        getattr(types, "not_a_dtype")
