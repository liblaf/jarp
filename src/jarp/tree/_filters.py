from collections.abc import Iterable
from typing import Any

import jax
from jax import Array

from ._define import frozen

type Leaf = Any
type PyTreeDef[T] = Any


@frozen(static=True)
class AuxData[T]:
    meta_fields: tuple[Any, ...]
    treedef: PyTreeDef[T]


def combine[T](data_leaves: Iterable[Any], aux: AuxData[T]) -> T:
    leaves: list[Any] = combine_leaves(data_leaves, aux.meta_fields)
    return jax.tree.unflatten(aux.treedef, leaves)


def combine_leaves(data_leaves: Iterable[Any], meta_leaves: Iterable[Any]) -> list[Any]:
    return [
        meta_leaf if data_leaf is None else data_leaf
        for data_leaf, meta_leaf in zip(data_leaves, meta_leaves, strict=True)
    ]


def is_array(obj: Any) -> bool:
    return isinstance(obj, Array)


def partition[T](obj: T) -> tuple[list[Any], AuxData[T]]:
    leaves: list[Any]
    treedef: PyTreeDef[T]
    leaves, treedef = jax.tree.flatten(obj)
    data_leaves: list[Any]
    meta_leaves: list[Any]
    data_leaves, meta_leaves = partition_leaves(leaves)
    aux: AuxData = AuxData(tuple(meta_leaves), treedef)
    return data_leaves, aux


def partition_leaves(leaves: Iterable[Any]) -> tuple[list[Any], list[Any]]:
    data_leaves: list[Any] = []
    meta_leaves: list[Any] = []
    for leaf in leaves:
        if is_array(leaf):
            data_leaves.append(leaf)
            meta_leaves.append(None)
        else:
            data_leaves.append(None)
            meta_leaves.append(leaf)
    return data_leaves, meta_leaves
