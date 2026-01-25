import functools
import types
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Protocol, Self, TypedDict, Unpack, overload

import jax

from jarp import tree

type _Dynamic = Iterable[Any]
type _Static[T] = tree.AuxData[T]


class JitOptions(TypedDict, total=False):
    filter: bool
    inline: bool


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

    fun_dynamic: _Dynamic
    fun_static: _Static[Callable[P, T]]
    fun_dynamic, fun_static = tree.partition(fun)
    inner: _Inner[T] = _Inner(fun_static)
    inner_jit: _InnerProtocol[T] = jax.jit(inner, static_argnums=(2,), **kwargs)
    outer: _Outer[P, T] = _Outer(fun_dynamic=fun_dynamic, inner=inner_jit)
    functools.update_wrapper(outer, fun)
    return outer


class _InnerProtocol[T](Protocol):
    def __call__(
        self, inputs_dynamic: _Dynamic, fun_dynamic: _Dynamic, inputs_static: _Static
    ) -> tuple[_Dynamic, _Static[T]]: ...


@tree.frozen(static=True)
class _Inner[T](_InnerProtocol[T]):
    fun_static: _Static[Callable[..., T]]

    def __call__(
        self, inputs_dynamic: _Dynamic, fun_dynamic: _Dynamic, inputs_static: _Static
    ) -> tuple[_Dynamic, _Static[T]]:
        fun: Callable[..., T] = tree.combine(fun_dynamic, self.fun_static)
        args: Sequence[Any]
        kwargs: Mapping[str, Any]
        args, kwargs = tree.combine(inputs_dynamic, inputs_static)
        outputs: T = fun(*args, **kwargs)
        outputs_dynamic: _Dynamic
        outputs_static: _Static[T]
        outputs_dynamic, outputs_static = tree.partition(outputs)
        return outputs_dynamic, outputs_static


@tree.define(slots=False, static=False)
class _Outer[**P, T]:
    fun_dynamic: _Dynamic
    inner: _InnerProtocol[T]

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        inputs_dynamic: _Dynamic
        inputs_static: _Static
        inputs_dynamic, inputs_static = tree.partition((args, kwargs))
        outputs_dynamic: _Dynamic
        outputs_static: _Static[T]
        outputs_dynamic, outputs_static = self.inner(
            inputs_dynamic, self.fun_dynamic, inputs_static
        )
        return tree.combine(outputs_dynamic, outputs_static)

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
