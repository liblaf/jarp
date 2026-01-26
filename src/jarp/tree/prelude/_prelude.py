import functools

from ._method import register_pytree_method
from ._warp import register_warp_array


@functools.cache  # run only once
def register_pytree_prelude() -> None:
    register_pytree_method()
    register_warp_array()
