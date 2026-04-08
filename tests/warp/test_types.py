import pytest
import warp as wp
from warp._src.types import type_scalar_type

from jarp.warp import types as wt


def test_dynamic_type_helpers_follow_the_active_jax_precision() -> None:
    assert wt.floating is wp.float64
    assert type_scalar_type(wt.vector(3)) is wp.float64
    assert type_scalar_type(wt.matrix((2, 3))) is wp.float64
    assert wp.types.types_equal(wt.vec3, wt.vector(3))
    assert wp.types.types_equal(wt.mat23, wt.matrix((2, 3)))


def test_dynamic_attribute_lookup_handles_deprecations_and_errors() -> None:
    with pytest.warns(DeprecationWarning, match="deprecated"):
        assert wt.float is wt.floating

    with pytest.raises(AttributeError):
        _ = wt.missing
