from collections.abc import Iterable
from typing import Any

import jax

from jarp.utils import is_array

from ._define import frozen

type Leaf = Any
type PyTreeDef = Any


@frozen(static=True)
class AuxData[T]:
    static_fields: tuple[Any, ...]
    treedef: PyTreeDef[T]


def combine[T](dynamic_leaves: Iterable[Any], aux: AuxData[T]) -> T:
    leaves: list[Any] = combine_leaves(dynamic_leaves, aux.static_fields)
    return jax.tree.unflatten(aux.treedef, leaves)


def combine_leaves(
    dynamic_leaves: Iterable[Any], static_leaves: Iterable[Any]
) -> list[Any]:
    return [
        static_leaf if dynamic_leaf is None else dynamic_leaf
        for dynamic_leaf, static_leaf in zip(dynamic_leaves, static_leaves, strict=True)
    ]


def partition[T](obj: T) -> tuple[list[Any], AuxData[T]]:
    leaves: list[Any]
    treedef: PyTreeDef[T]
    leaves, treedef = jax.tree.flatten(obj)
    dynamic_leaves: list[Any]
    static_leaves: list[Any]
    dynamic_leaves, static_leaves = partition_leaves(leaves)
    aux: AuxData = AuxData(tuple(static_leaves), treedef)
    return dynamic_leaves, aux


def partition_leaves(leaves: Iterable[Any]) -> tuple[list[Any], list[Any]]:
    dynamic_leaves: list[Any] = []
    static_leaves: list[Any] = []
    for leaf in leaves:
        if is_array(leaf):
            dynamic_leaves.append(leaf)
            static_leaves.append(None)
        else:
            dynamic_leaves.append(None)
            static_leaves.append(leaf)
    return dynamic_leaves, static_leaves
