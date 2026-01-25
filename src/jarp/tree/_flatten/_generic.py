from collections.abc import Sequence
from typing import Any, cast

import jax.tree_util as jtu
from jax import Array
from jax._src.tree_util import _registry

from ._types import (
    FlattenFunction,
    FlattenWithKeysFunction,
    PyTreeFunctions,
    UnflattenFunction,
)

type _KeyEntry = Any


def is_pytree_node(obj: Any) -> bool:
    return isinstance(obj, Array) or type(obj) in _registry


def make_pytree_functions[T](
    cls: type[T],
    data_fields: Sequence[str],
    meta_fields: Sequence[str],
    unknown_fields: Sequence[str],
) -> PyTreeFunctions[T]:
    return PyTreeFunctions(
        _make_flatten(cls, data_fields, meta_fields, unknown_fields),
        _make_flatteen_with_keys(cls, data_fields, meta_fields, unknown_fields),
        _make_unflatten(cls, data_fields, meta_fields, unknown_fields),
    )


def register_generic[T](
    cls: type[T],
    data_fields: Sequence[str] = (),
    meta_fields: Sequence[str] = (),
    unknown_fields: Sequence[str] = (),
) -> None:
    flatten: FlattenFunction[T]
    flatten_with_keys: FlattenWithKeysFunction[T]
    unflatten: UnflattenFunction[T]
    flatten, flatten_with_keys, unflatten = make_pytree_functions(
        cls, data_fields, meta_fields, unknown_fields
    )
    jtu.register_pytree_node(cls, flatten, unflatten, flatten_with_keys)


def _make_flatten[T](
    _cls: type[T],
    data_fields: Sequence[str],
    meta_fields: Sequence[str],
    unknown_fields: Sequence[str],
) -> FlattenFunction[T]:
    def flatten(obj: T) -> tuple[list[Any], tuple[Any, ...]]:
        data_leaves: list[Any] = [getattr(obj, name) for name in data_fields]
        meta_leaves: list[Any] = [getattr(obj, name) for name in meta_fields]
        for name in unknown_fields:
            leaf: Any = getattr(obj, name)
            if is_pytree_node(leaf):
                data_leaves.append(leaf)
                meta_leaves.append(None)
            else:
                data_leaves.append(None)
                meta_leaves.append(leaf)
        return data_leaves, tuple(meta_leaves)

    return flatten


def _make_flatteen_with_keys[T](
    _cls: type[T],
    data_fields: Sequence[str],
    meta_fields: Sequence[str],
    unknown_fields: Sequence[str],
) -> FlattenWithKeysFunction[T]:
    def tree_flatten_with_keys(
        obj: T,
    ) -> tuple[list[tuple[_KeyEntry, Any]], tuple[Any, ...]]:
        data_leaves: list[tuple[_KeyEntry, Any]] = [
            (jtu.GetAttrKey(name), getattr(obj, name)) for name in data_fields
        ]
        meta_leaves: list[Any] = [getattr(obj, name) for name in meta_fields]
        for name in unknown_fields:
            leaf: Any = getattr(obj, name)
            if is_pytree_node(leaf):
                data_leaves.append((jtu.GetAttrKey(name), leaf))
                meta_leaves.append(None)
            else:
                data_leaves.append((jtu.GetAttrKey(name), None))
                meta_leaves.append(leaf)
        return data_leaves, tuple(meta_leaves)

    return tree_flatten_with_keys


def _make_unflatten[T](
    cls: type[T],
    data_fields: Sequence[str],
    meta_fields: Sequence[str],
    unknown_fields: Sequence[str],
) -> UnflattenFunction[T]:
    def unflatten(aux: Sequence[Any], children: Sequence[Any]) -> T:
        obj: T = object.__new__(cls)
        for name, value in zip(data_fields, children, strict=False):
            object.__setattr__(obj, name, value)
        for name, value in zip(meta_fields, aux, strict=False):
            object.__setattr__(obj, name, value)
        for name, data_leaf, meta_leaf in zip(
            unknown_fields,
            children[len(data_fields) :],
            aux[len(meta_fields) :],
            strict=True,
        ):
            object.__setattr__(obj, name, meta_leaf if data_leaf is None else data_leaf)
        return obj

    return cast("UnflattenFunction[T]", unflatten)
