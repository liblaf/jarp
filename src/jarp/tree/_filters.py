from collections.abc import Iterable
from typing import Any

import jax
import jax.tree_util as jtu
from jax import Array
from jax._src.tree_util import _registry
from liblaf import grapes
from typing_extensions import TypeIs


@jtu.register_static
@grapes.attrs.frozen
class AuxData[T]:
    meta_leaves: tuple[Any, ...]
    treedef: Any


def is_data(obj: Any) -> bool:
    return obj is None or isinstance(obj, Array) or type(obj) in _registry


def is_leaf(obj: Any) -> TypeIs[Array | None]:
    return obj is None or isinstance(obj, Array)


def combine[T](data_leaves: Iterable[Array | None], aux: AuxData[T]) -> T:
    leaves: list[Any] = combine_leaves(data_leaves, aux.meta_leaves)
    return jax.tree.unflatten(aux.treedef, leaves)


def combine_leaves(
    data_leaves: Iterable[Array | None], meta_leaves: Iterable[Any]
) -> list[Any]:
    return [
        data_leaf if meta_leaf is None else meta_leaf
        for data_leaf, meta_leaf in zip(data_leaves, meta_leaves, strict=True)
    ]


def partition(obj: Any) -> tuple[list[Array | None], AuxData]:
    leaves: list[Any]
    treedef: Any
    leaves, treedef = jax.tree.flatten(obj)
    data_leaves: list[Array | None]
    meta_leaves: list[Any]
    data_leaves, meta_leaves = partition_leaves(leaves)
    return data_leaves, AuxData(tuple(meta_leaves), treedef)


def partition_leaves(leaves: list[Any]) -> tuple[list[Array | None], list[Any]]:
    data_leaves: list[Array | None] = []
    meta_leaves: list[Any] = []
    for leaf in leaves:
        if is_leaf(leaf):
            data_leaves.append(leaf)
            meta_leaves.append(None)
        else:
            data_leaves.append(None)
            meta_leaves.append(leaf)
    return data_leaves, meta_leaves
