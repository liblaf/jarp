from collections.abc import Iterable

import jax.tree_util as jtu
import wrapt
from jax import Array

from jarp.tree._filters import AuxData, combine, partition


class PyTreeProxy[T](wrapt.BaseObjectProxy):
    __wrapped__: T


def _proxy_flatten[T](obj: PyTreeProxy[T]) -> tuple[list[Array | None], AuxData[T]]:
    return partition(obj.__wrapped__)


def _proxy_unflatten[T](
    aux: AuxData[T], children: Iterable[Array | None]
) -> PyTreeProxy[T]:
    return PyTreeProxy(combine(children, aux))


jtu.register_pytree_node(PyTreeProxy, _proxy_flatten, _proxy_unflatten)
