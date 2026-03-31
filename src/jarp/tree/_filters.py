from collections.abc import Iterable
from typing import Any

import attrs
import jax
import jax.tree_util as jtu
from jax import Array
from jax._src.tree_util import _registry
from typing_extensions import TypeIs


@jtu.register_static
@attrs.frozen
class AuxData[T]:
    """Carry the static part of a partitioned PyTree."""

    meta_leaves: tuple[Any, ...]
    treedef: Any


def is_data(obj: Any) -> bool:
    """Return whether an object should stay on the dynamic side of a partition."""
    return obj is None or isinstance(obj, Array) or type(obj) in _registry


def is_leaf(obj: Any) -> TypeIs[Array | None]:
    """Return whether a leaf contributes data to a flattened vector."""
    return obj is None or isinstance(obj, Array)


def combine[T](data_leaves: Iterable[Array | None], aux: AuxData[T]) -> T:
    """Rebuild a PyTree from dynamic leaves and recorded auxiliary data."""
    leaves: list[Any] = combine_leaves(data_leaves, aux.meta_leaves)
    return jax.tree.unflatten(aux.treedef, leaves)


def combine_leaves(
    data_leaves: Iterable[Array | None], meta_leaves: Iterable[Any]
) -> list[Any]:
    """Zip dynamic leaves back together with their static counterparts."""
    return [
        data_leaf if meta_leaf is None else meta_leaf
        for data_leaf, meta_leaf in zip(data_leaves, meta_leaves, strict=True)
    ]


def partition[T](obj: T) -> tuple[list[Array | None], AuxData[T]]:
    """Split a PyTree into dynamic leaves and static metadata."""
    leaves: list[Any]
    treedef: Any
    leaves, treedef = jax.tree.flatten(obj)
    data_leaves: list[Array | None]
    meta_leaves: list[Any]
    data_leaves, meta_leaves = partition_leaves(leaves)
    return data_leaves, AuxData(tuple(meta_leaves), treedef)


def partition_leaves(leaves: list[Any]) -> tuple[list[Array | None], list[Any]]:
    """Separate raw tree leaves into data leaves and metadata leaves."""
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
