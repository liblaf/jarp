from collections.abc import Hashable, Iterable
from typing import Any, NamedTuple, Protocol

type KeyEntry = Any


class FlattenFunction[T](Protocol):
    def __call__(self, obj: T) -> tuple[Iterable[Any], Hashable]: ...


class FlattenWithKeysFunction[T](Protocol):
    def __call__(self, obj: T) -> tuple[Iterable[tuple[KeyEntry, Any]], Hashable]: ...


class UnflattenFunction[T](Protocol):
    def __call__(self, aux: Hashable, children: Iterable[Any]) -> T: ...


class PyTreeFunctions[T](NamedTuple):
    tree_flatten: FlattenFunction[T]
    tree_flatten_with_keys: FlattenWithKeysFunction[T]
    tree_unflatten: UnflattenFunction[T]
