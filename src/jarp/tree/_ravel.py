from collections.abc import Iterable
from typing import Any, cast

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, DTypeLike, Shaped

from ._filters import combine_leaves, partition_leaves
from .attrs import frozen_static

type PyTreeDef = Any
type Shape = tuple[int, ...]
type Vector = Shaped[Array, " N"]


@frozen_static
class Structure[T]:
    """Record how to flatten and rebuild a PyTree's dynamic leaves.

    Instances are returned by :func:`ravel` and capture the original tree
    definition, the static leaves that were removed from the flat vector, and
    the offsets needed to reconstruct each dynamic leaf.
    """

    dtype: DTypeLike
    meta_leaves: tuple[Any, ...]
    offsets: tuple[int, ...]
    shapes: tuple[Shape | None, ...]
    treedef: PyTreeDef

    @property
    def is_leaf(self) -> bool:
        """Return whether the original tree was a single leaf."""
        return jtu.treedef_is_leaf(self.treedef)

    def ravel(self, tree: T | Array) -> Vector:
        """Flatten a compatible tree or flatten an array in-place.

        Args:
            tree: A tree with the same structure used to build this
                :class:`Structure`, or an already-flat array.

        Returns:
            A one-dimensional array containing the dynamic leaves.
        """
        if isinstance(tree, Array):
            # do not flatten if already flat
            return jnp.ravel(tree)
        leaves, treedef = jax.tree.flatten(tree)
        assert treedef == self.treedef
        data_leaves, meta_leaves = partition_leaves(leaves)
        assert tuple(meta_leaves) == self.meta_leaves
        return _ravel(data_leaves)

    def unravel(self, flat: T | Array, dtype: DTypeLike | None = None) -> T:
        """Rebuild the original tree shape from a flat vector.

        Args:
            flat: One-dimensional data produced by :meth:`ravel`, or a tree that
                already matches the recorded structure.
            dtype: Optional dtype override applied to the flat array before it
                is split and reshaped.

        Returns:
            A tree with the same structure and static metadata as the original
            input to :func:`ravel`.
        """
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
    """Flatten a PyTree's dynamic leaves into one vector.

    Non-array leaves are treated as static metadata and preserved in the
    returned :class:`Structure` instead of being concatenated into the flat
    array.

    Args:
        tree: PyTree to flatten.

    Returns:
        A tuple of ``(flat, structure)`` where ``flat`` is a one-dimensional
        JAX array and ``structure`` can rebuild compatible trees later.
    """
    leaves, treedef = jax.tree.flatten(tree)
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
    leaves: list[Array] = [leaf for leaf in leaves if leaf is not None]
    if not leaves:
        return jnp.empty((0,), dtype=jnp.float32)
    return jnp.concatenate(leaves, axis=None)


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
