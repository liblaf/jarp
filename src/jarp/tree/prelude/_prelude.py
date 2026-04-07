import functools

from ._method import register_pytree_method
from ._warp import register_warp_array


@functools.cache  # run only once
def register_pytree_prelude() -> None:
    """Register the built-in PyTree adapters used by jarp.

    This function is idempotent. It currently registers bound methods and
    ``warp.array`` so they participate correctly in tree traversals.
    """
    register_pytree_method()
    register_warp_array()
