import functools

from ._method import register_pytree_method


@functools.cache  # run only once
def register_pytree_prelude() -> None:
    register_pytree_method()
