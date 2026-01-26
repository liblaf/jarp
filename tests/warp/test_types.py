import jax
import warp as wp

import jarp.warp.types as wpt


def test_warp_types() -> None:
    with jax.enable_x64(True):  # noqa: FBT003
        assert wpt.float_ is wp.float64
        assert wpt.vec2 is wp.types.vector(2, wp.float64)
        assert wpt.vec3 is wp.types.vector(3, wp.float64)
        assert wpt.vec4 is wp.types.vector(4, wp.float64)
        assert wpt.mat22 is wp.types.matrix((2, 2), wp.float64)
        assert wpt.mat33 is wp.types.matrix((3, 3), wp.float64)
        assert wpt.mat44 is wp.types.matrix((4, 4), wp.float64)
        assert wpt.vector(5) is wp.types.vector(5, wp.float64)
        assert wpt.matrix((2, 3)) is wp.types.matrix((2, 3), wp.float64)
    with jax.enable_x64(False):  # noqa: FBT003
        assert wpt.float_ is wp.float32
        assert wpt.vec2 is wp.types.vector(2, wp.float32)
        assert wpt.vec3 is wp.types.vector(3, wp.float32)
        assert wpt.vec4 is wp.types.vector(4, wp.float32)
        assert wpt.mat22 is wp.types.matrix((2, 2), wp.float32)
        assert wpt.mat33 is wp.types.matrix((3, 3), wp.float32)
        assert wpt.mat44 is wp.types.matrix((4, 4), wp.float32)
        assert wpt.vector(5) is wp.types.vector(5, wp.float32)
        assert wpt.matrix((2, 3)) is wp.types.matrix((2, 3), wp.float32)
