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


def is_data_leaf(obj: Any) -> bool:
    return obj is None or isinstance(obj, Array)


def partition[T](obj: T) -> tuple[list[Any], AuxData[T]]:
    leaves: list[Any]
    treedef: PyTreeDef[T]
    leaves, treedef = jax.tree.flatten(obj)
    data_leaves: list[Any]
    meta_leaves: list[Any]
    data_leaves, meta_leaves = partition_leaves(leaves)
    aux: AuxData = AuxData(tuple(meta_leaves), treedef)
    return data_leaves, aux


def partition_with_path[T](obj: T) -> tuple[list[tuple[Any, object]], AuxData[T]]:
    leaves_with_path: list[tuple[Any, Any]]
    treedef: PyTreeDef[T]
    leaves_with_path, treedef = jax.tree.flatten_with_path(obj)
    data_leaves_with_path: list[tuple[Any, object]]
    meta_leaves: list[Any]
    data_leaves_with_path, meta_leaves = partition_leaves_with_path(leaves_with_path)
    aux: AuxData = AuxData(tuple(meta_leaves), treedef)
    return data_leaves_with_path, aux


def partition_leaves(leaves: Iterable[Any]) -> tuple[list[Any], list[Any]]:
    data_leaves: list[Any] = []
    meta_leaves: list[Any] = []
    for leaf in leaves:
        if is_data_leaf(leaf):
            data_leaves.append(leaf)
            meta_leaves.append(None)
        else:
            data_leaves.append(None)
            meta_leaves.append(leaf)
    return data_leaves, meta_leaves


def partition_leaves_with_path(
    leaves_with_path: Iterable[tuple[Any, Any]],
) -> tuple[list[tuple[Any, Any]], list[Any]]:
    data_leaves_with_path: list[tuple[Any, object]] = []
    meta_leaves: list[Any] = []
    for path, leaf in leaves_with_path:
        if is_data_leaf(leaf):
            data_leaves_with_path.append((path, leaf))
            meta_leaves.append(None)
        else:
            data_leaves_with_path.append((path, None))
            meta_leaves.append(leaf)
    return data_leaves_with_path, meta_leaves
