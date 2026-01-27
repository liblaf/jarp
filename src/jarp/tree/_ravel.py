from collections.abc import Iterable
from typing import Any, cast

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, ArrayLike, DTypeLike, Shaped

from ._filters import combine_leaves, partition_leaves
from .attrs import frozen_static

type PyTreeDef = Any
type Shape = tuple[int, ...]
type Vector = Shaped[Array, " N"]


@frozen_static
class Structure[T]:
    dtype: DTypeLike
    meta_leaves: tuple[Any, ...]
    offsets: tuple[int, ...]
    shapes: tuple[Shape | None, ...]
    treedef: PyTreeDef

    @property
    def is_leaf(self) -> bool:
        return jtu.treedef_is_leaf(self.treedef)

    def ravel(self, tree: T | Array) -> Vector:
        if isinstance(tree, Array):
            # do not flatten if already flat
            return jnp.ravel(tree)
        leaves: list[Any]
        treedef: PyTreeDef
        leaves, treedef = jax.tree.flatten(tree)
        assert treedef == self.treedef
        data_leaves: list[Array | None]
        meta_leaves: list[Any]
        data_leaves, meta_leaves = partition_leaves(leaves)
        assert meta_leaves == self.meta_leaves
        return _ravel(data_leaves)

    def unravel(self, flat: T | ArrayLike, dtype: DTypeLike | None = None) -> T:
        if not isinstance(flat, Array):
            # do not unravel if already a pytree
            assert jax.tree.structure(flat) == self.treedef
            return cast("T", flat)
        flat: Array = jnp.asarray(flat, self.dtype if dtype is None else dtype)
        if self.is_leaf:
            return cast("T", jnp.reshape(flat, self.shapes[0]))
        data_leaves: list[Array | None] = _unravel(flat, self.offsets, self.shapes)
        leaves: list[Any] = combine_leaves(data_leaves, self.meta_leaves)
        return jax.tree.unflatten(self.treedef, leaves)


def ravel[T](tree: T) -> tuple[Array, Structure[T]]:
    leaves: list[Any]
    treedef: PyTreeDef
    leaves, treedef = jax.tree.flatten(tree)
    dynamic_leaves: list[Any | None]
    static_leaves: list[Any | None]
    dynamic_leaves, static_leaves = partition_leaves(leaves)
    flat: Array = _ravel(dynamic_leaves)
    structure: Structure[T] = Structure(
        offsets=_offsets_from_leaves(dynamic_leaves),
        shapes=_shapes_from_leaves(dynamic_leaves),
        meta_leaves=tuple(static_leaves),
        treedef=treedef,
        dtype=flat.dtype,
    )
    return flat, structure


def _offsets_from_leaves(leaves: Iterable[Any | None]) -> tuple[int, ...]:
    offsets: list[int] = []
    i: int = 0
    for leaf in leaves:
        if leaf is not None:
            i += jnp.size(leaf)
        offsets.append(i)
    return tuple(offsets)


@jax.jit
def _ravel(leaves: Iterable[Array | None]) -> Array:
    return jnp.concatenate([leaf for leaf in leaves if leaf is not None], axis=None)


def _shapes_from_leaves(leaves: Iterable[Any | None]) -> tuple[Shape | None, ...]:
    return tuple(None if leaf is None else jnp.shape(leaf) for leaf in leaves)


@jax.jit(static_argnums=(1, 2))
def _unravel(
    flat: Array, offsets: tuple[int, ...], shapes: tuple[Shape | None, ...]
) -> list[Array | None]:
    assert jnp.size(flat) == offsets[-1]
    chunks: list[Array] = jnp.split(flat, offsets[:-1])
    leaves: list[Array | None] = []
    for chunk, shape in zip(chunks, shapes, strict=True):
        if shape is None:
            assert jnp.size(chunk) == 0
            leaves.append(None)
        else:
            leaves.append(jnp.reshape(chunk, shape))
    return leaves
