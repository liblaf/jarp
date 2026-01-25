from typing import Any

import jax.tree_util as jtu
import wrapt

from jarp.tree._filters import AuxData, combine, partition, partition_with_path


class PyTreeProxy[T](wrapt.BaseObjectProxy):
    __wrapped__: T


def _proxy_flatten[T](obj: PyTreeProxy[T]) -> tuple[list[Any], AuxData[T]]:
    return partition(obj.__wrapped__)


def _proxy_flatten_with_keys[T](
    obj: PyTreeProxy[T],
) -> tuple[list[tuple[Any, Any]], AuxData[T]]:
    return partition_with_path(obj.__wrapped__)


def _proxy_unflatten[T](aux: AuxData[T], children: list[Any]) -> PyTreeProxy[T]:
    return PyTreeProxy(combine(children, aux))


jtu.register_pytree_node(
    PyTreeProxy, _proxy_flatten, _proxy_unflatten, _proxy_flatten_with_keys
)
