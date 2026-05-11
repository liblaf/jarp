from __future__ import annotations

import functools
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol, Self, TypedDict, Unpack, overload

import jax
from jaxtyping import Array

from liblaf.jarp import tree

if TYPE_CHECKING:
    from _typeshed import IdentityFunction


class InnerLike[**P, T](Protocol):
    def __call__(
        self,
        fun_data: Iterable[Array | None],
        inputs_data: Iterable[Array | None],
        inputs_meta: tree.AuxData[tuple[Sequence[Any], Mapping[str, Any]]],
    ) -> tuple[Sequence[Array | None], tree.AuxData[T]]: ...


@tree.frozen_static
class FilterInner[**P, T]:
    fun_meta: tree.AuxData[Callable[P, T]] = tree.static()

    def __call__(
        self,
        fun_data: Iterable[Array | None],
        inputs_data: Iterable[Array | None],
        inputs_meta: tree.AuxData[tuple[Sequence[Any], Mapping[str, Any]]],
    ) -> tuple[list[Array | None], tree.AuxData[T]]:
        fun: Callable[P, T] = tree.combine(fun_data, self.fun_meta)
        args, kwargs = tree.combine(inputs_data, inputs_meta)
        output: T = fun(*args, **kwargs)
        output_data, output_meta = tree.partition(output)
        return output_data, output_meta


@tree.define(slots=False)
class FilterOuter[**P, T]:
    fun_data: list[Array | None] = tree.field()
    inner: InnerLike[P, T] = tree.static()

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        inputs_data, inputs_meta = tree.partition((args, kwargs))
        output_data, output_meta = self.inner(self.fun_data, inputs_data, inputs_meta)
        return tree.combine(output_data, output_meta)

    @overload
    def __get__(self, instance: None, owner: type | None = None) -> Self: ...
    @overload
    def __get__(
        self, instance: object, owner: type | None = None
    ) -> Callable[..., T]: ...
    def __get__(
        self, instance: object | None, owner: type | None = None
    ) -> Self | Callable[..., T]:
        if instance is None:
            return self
        return tree.partial(self, instance)


class FilterJitOptions(TypedDict, total=False):
    keep_unused: bool
    device: Any | None
    backend: str | None
    inline: bool


@overload
def filter_jit[F: Callable[..., Any]](
    fun: F, **kwargs: Unpack[FilterJitOptions]
) -> F: ...
@overload
def filter_jit(
    fun: None = None, **kwargs: Unpack[FilterJitOptions]
) -> IdentityFunction: ...
def filter_jit[F: Callable[..., Any]](
    fun: F | None = None, **kwargs: Unpack[FilterJitOptions]
) -> Callable[..., Any]:
    """Wrap a callable with `liblaf.jarp` data-versus-metadata partitioning.

    The wrapper partitions the callable and each invocation's arguments with
    [`partition`][liblaf.jarp.tree.partition], rebuilds the original call shape,
    and partitions the return value again before handing it back. This keeps
    JAX arrays on the dynamic side of the partition while preserving ordinary
    Python metadata such as strings, bound methods, or configuration objects.

    Args:
        fun: Callable to wrap. When omitted, return a configured decorator.
        **kwargs: Reserved compatibility options for a `jax.jit`-like
            surface. The current implementation accepts these names but does
            not use them directly.

    Returns:
        The wrapped callable, or a decorator that produces one.
    """
    if fun is None:
        return functools.partial(filter_jit, **kwargs)
    fun_data, fun_meta = tree.partition(fun)
    inner: FilterInner = FilterInner(fun_meta=fun_meta)
    inner_jit: InnerLike = jax.jit(inner, **kwargs)
    outer: FilterOuter = FilterOuter(fun_data=fun_data, inner=inner_jit)
    functools.update_wrapper(outer, fun)
    return outer
