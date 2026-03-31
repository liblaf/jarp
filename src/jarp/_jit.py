import functools
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Literal, Protocol, Self, TypedDict, Unpack, overload

import jax
import jax.tree_util as jtu

from jarp import tree

type _Data = Iterable[jax.Array | None]
type _Meta[T] = tree.AuxData[T]


class JitOptions(TypedDict, total=False):
    in_shardings: Any
    out_shardings: Any
    static_argnums: int | Sequence[int] | None
    static_argnames: str | Iterable[str] | None
    donate_argnums: int | Sequence[int] | None
    donate_argnames: str | Iterable[str] | None
    keep_unused: bool
    device: Any | None
    backend: str | None
    inline: bool
    compiler_options: dict[str, Any] | None


class FilterJitOptions(TypedDict, total=False):
    keep_unused: bool
    device: Any | None
    backend: str | None
    inline: bool
    compiler_options: dict[str, Any] | None


@overload
def jit[**P, T](
    fun: Callable[P, T],
    /,
    *,
    filter: Literal[False] = False,
    **kwargs: Unpack[JitOptions],
) -> Callable[P, T]: ...
@overload
def jit[**P, T](
    *, filter: Literal[False] = False, **kwargs: Unpack[JitOptions]
) -> Callable[[Callable[P, T]], Callable[P, T]]: ...
@overload
def jit[**P, T](
    fun: Callable[P, T], /, *, filter: Literal[True], **kwargs: Unpack[FilterJitOptions]
) -> Callable[P, T]: ...
@overload
def jit[**P, T](
    *, filter: Literal[True], **kwargs: Unpack[FilterJitOptions]
) -> Callable[[Callable[P, T]], Callable[P, T]]: ...
def jit[**P, T](fun: Callable[P, T] | None = None, **kwargs: Any) -> Callable:
    """Compile a callable with JAX, optionally preserving static PyTree leaves.

    When ``filter=False`` this is a thin wrapper around :func:`jax.jit`.
    When ``filter=True`` the function and its inputs are partitioned into
    dynamic array leaves and static metadata so mixed PyTrees can cross the JIT
    boundary without requiring manual ``static_argnums`` wiring.

    Args:
        fun: Callable to compile. When omitted, return a decorator.
        **kwargs: Options forwarded to :func:`jax.jit`. With ``filter=True``,
            only the subset in :class:`FilterJitOptions` is supported because
            static argument handling is managed internally.

    Returns:
        A compiled callable or decorator with the same public call signature as
        ``fun``.
    """
    if fun is None:
        return functools.partial(jit, **kwargs)
    filter_: bool = kwargs.pop("filter", False)
    if not filter_:
        return jax.jit(fun, **kwargs)

    fun_data: _Data
    fun_meta: _Meta[Callable[P, T]]
    fun_data, fun_meta = tree.partition(fun)
    inner: _Inner[T] = _Inner(fun_meta)
    inner_jit: _InnerProtocol[T] = jax.jit(inner, static_argnums=(2,), **kwargs)
    outer: _Outer[P, T] = _Outer(fun_data=fun_data, inner=inner_jit)
    functools.update_wrapper(outer, fun)
    return outer


class _InnerProtocol[T](Protocol):
    def __call__(
        self, inputs_data: _Data, fun_data: _Data, inputs_meta: _Meta, /
    ) -> tuple[_Data, _Meta[T]]: ...


@tree.frozen_static
class _Inner[T](_InnerProtocol[T]):
    """Rebuild the original callable inside the compiled function body."""

    fun_meta: _Meta[Callable[..., T]]

    def __call__(
        self, inputs_data: _Data, fun_data: _Data, inputs_meta: _Meta
    ) -> tuple[_Data, _Meta[T]]:
        fun: Callable[..., T] = tree.combine(fun_data, self.fun_meta)
        args: Sequence[Any]
        kwargs: Mapping[str, Any]
        args, kwargs = tree.combine(inputs_data, inputs_meta)
        outputs: T = fun(*args, **kwargs)
        outputs_data: _Data
        outputs_meta: _Meta[T]
        outputs_data, outputs_meta = tree.partition(outputs)
        return outputs_data, outputs_meta


@tree.define(slots=False)
class _Outer[**P, T]:
    """Bind static function metadata around the compiled inner callable."""

    fun_data: _Data
    inner: _InnerProtocol[T]

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        inputs_data: _Data
        inputs_meta: _Meta
        inputs_data, inputs_meta = tree.partition((args, kwargs))
        outputs_data: _Data
        outputs_meta: _Meta[T]
        outputs_data, outputs_meta = self.inner(inputs_data, self.fun_data, inputs_meta)
        return tree.combine(outputs_data, outputs_meta)

    @overload
    def __get__(self, instance: None, owner: type | None = None) -> Self: ...
    @overload
    def __get__(
        self, instance: object, owner: type | None = None
    ) -> Callable[..., T]: ...
    def __get__(self, instance: Any, owner: type | None = None) -> Callable[..., T]:
        if instance is None:
            return self
        return jtu.Partial(self, instance)


tree.register_pytree_prelude()
