import functools
import types
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Protocol, Self, TypedDict, Unpack, overload

import jax

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

    filter: bool


@overload
def jit[**P, T](
    fun: Callable[P, T], **kwargs: Unpack[JitOptions]
) -> Callable[P, T]: ...
@overload
def jit[**P, T](
    **kwargs: Unpack[JitOptions],
) -> Callable[[Callable[P, T]], Callable[P, T]]: ...
def jit[**P, T](fun: Callable[P, T] | None = None, **kwargs) -> Callable:
    if fun is None:
        return functools.partial(jit, **kwargs)
    filter_: bool = kwargs.pop("filter", False)
    if not filter_:
        return jax.jit(fun, **kwargs)

    assert "static_argnums" not in kwargs
    assert "static_argnames" not in kwargs
    assert "donate_argnums" not in kwargs
    assert "donate_argnames" not in kwargs

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
        self, inputs_dynamic: _Data, fun_dynamic: _Data, inputs_static: _Meta
    ) -> tuple[_Data, _Meta[T]]: ...


@tree.frozen_static
class _Inner[T](_InnerProtocol[T]):
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
        return types.MethodType(self, instance)


tree.register_pytree_prelude()
