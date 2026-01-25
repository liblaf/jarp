import functools
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Literal, Protocol, TypedDict, Unpack, overload

import jax.tree_util as jtu
import warp.jax_experimental
from jax import Array
from warp._src.jax_experimental.ffi import FfiCallable as _WarpFfiCallable
from warp._src.jax_experimental.ffi import ModulePreloadMode
from warp.jax_experimental import GraphMode

from jarp import tree

from ._types import ShapeLike, VmapMethod, WarpScalarDType
from ._utils import dtypes_from_args

type _FfiCallableFunction = Callable[..., None]
type _FfiCallableFactory = Callable[..., _FfiCallableFunction]


class JaxCallableOptions(TypedDict, total=False):
    num_outputs: int
    graph_mode: GraphMode
    vmap_method: str | None
    output_dims: dict[str, ShapeLike] | None  # Mapping won't work with Warp
    in_out_argnames: Iterable[str]
    stage_in_argnames: Iterable[str]
    stage_out_argnames: Iterable[str]
    graph_cache_max: int | None
    module_preload_mode: ModulePreloadMode


class JaxCallableCallOptions(TypedDict, total=False):
    output_dims: ShapeLike | dict[str, ShapeLike] | None  # Mapping won't work with Warp
    vmap_method: VmapMethod | None


class FfiCallableProtocol(Protocol):
    def __call__(
        self, *args: Array, **kwargs: Unpack[JaxCallableCallOptions]
    ) -> Sequence[Array]: ...


@tree.frozen(static=True)
class _FfiCallable(FfiCallableProtocol):
    factory: _FfiCallableFactory
    options: JaxCallableOptions

    def __call__(
        self, *args: Array, **kwargs: Unpack[JaxCallableCallOptions]
    ) -> Sequence[Array]:
        warp_dtypes: list[WarpScalarDType] = dtypes_from_args(*args)
        func: _FfiCallableFunction = self.factory(*warp_dtypes)
        jax_callable: Callable[..., Sequence[Array]] = (
            warp.jax_experimental.jax_callable(func, **self.options)
        )
        return jax_callable(*args, **kwargs)


@overload
def jax_callable(
    func: _FfiCallableFunction,
    *,
    generic: Literal[False] = False,
    **kwargs: Unpack[JaxCallableOptions],
) -> FfiCallableProtocol: ...
@overload
def jax_callable(
    *, generic: Literal[False] = False, **kwargs: Unpack[JaxCallableOptions]
) -> Callable[[_FfiCallableFunction], FfiCallableProtocol]: ...
@overload
def jax_callable(
    func: _FfiCallableFactory,
    *,
    generic: Literal[True],
    **kwargs: Unpack[JaxCallableOptions],
) -> _FfiCallable: ...
@overload
def jax_callable(
    *, generic: Literal[True], **kwargs: Unpack[JaxCallableOptions]
) -> Callable[[_FfiCallableFactory], _FfiCallable]: ...
def jax_callable(
    func: Callable | None = None,
    *,
    generic: bool = False,
    **kwargs: Unpack[JaxCallableOptions],
) -> Any:
    if func is None:
        return functools.partial(jax_callable, generic=generic, **kwargs)
    if not generic:
        return warp.jax_experimental.jax_callable(func, **kwargs)
    return _FfiCallable(factory=func, options=kwargs)


jtu.register_static(_WarpFfiCallable)
