import functools
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Protocol, TypedDict, Unpack, cast, overload

import jax.tree_util as jtu
import warp as wp
import warp.jax_experimental
from jax import Array
from warp._src.jax_experimental.ffi import FfiKernel as _WarpFfiKernel
from warp._src.jax_experimental.ffi import ModulePreloadMode

from jarp import tree

from ._types import ArgTypes, ShapeLike, VmapMethod, WarpScalarDType
from ._utils import dtypes_from_args


class JaxKernelOptions(TypedDict, total=False):
    num_outputs: int
    vmap_method: VmapMethod
    launch_dims: ShapeLike | None
    output_dims: ShapeLike | dict[str, ShapeLike] | None  # Mapping won't work with Warp
    in_out_argnames: Iterable[str]
    module_preload_mode: ModulePreloadMode
    enable_backward: bool


class JaxKernelCallOptions(TypedDict, total=False):
    output_dims: ShapeLike | dict[str, ShapeLike] | None  # Mapping won't work with Warp
    launch_dims: ShapeLike | None
    vmap_method: VmapMethod | None


class FfiKernelProtocol(Protocol):
    def __call__(
        self, *args: Array, **kwargs: Unpack[JaxKernelCallOptions]
    ) -> Sequence[Array]: ...


@tree.frozen_static
class _FfiKernel(FfiKernelProtocol):
    kernel: wp.Kernel
    options: JaxKernelOptions
    arg_types_factory: Callable[..., ArgTypes]

    def __call__(
        self, *args: Array, **kwargs: Unpack[JaxKernelCallOptions]
    ) -> Sequence[Array]:
        warp_dtypes: list[WarpScalarDType] = dtypes_from_args(*args)
        arg_types: ArgTypes = self.arg_types_factory(*warp_dtypes)
        kernel: wp.Kernel = wp.overload(self.kernel, arg_types)
        jax_kernel: Callable[..., Sequence[Array]] = warp.jax_experimental.jax_kernel(
            kernel, **self.options
        )
        return jax_kernel(*args, **kwargs)


# we annotate `kernel` as `Callable` instead of `wp.Kernel` because `@wp.kernel`
# does not have good typing
@overload
def jax_kernel(
    *,
    arg_types_factory: ArgTypes | Callable[[WarpScalarDType], ArgTypes] | None = None,
    **kwargs: Unpack[JaxKernelOptions],
) -> Callable[[Callable], FfiKernelProtocol]: ...
@overload
def jax_kernel(
    kernel: Callable,
    *,
    arg_types_factory: ArgTypes | Callable[[WarpScalarDType], ArgTypes] | None = None,
    **kwargs: Unpack[JaxKernelOptions],
) -> FfiKernelProtocol: ...
def jax_kernel(
    kernel: Callable | None = None,
    *,
    arg_types_factory: ArgTypes | Callable[[WarpScalarDType], ArgTypes] | None = None,
    **kwargs: Unpack[JaxKernelOptions],
) -> Any:
    if kernel is None:
        return functools.partial(
            jax_kernel, arg_types_factory=arg_types_factory, **kwargs
        )
    if arg_types_factory is None:
        return warp.jax_experimental.jax_kernel(kernel, **kwargs)
    if not callable(arg_types_factory):
        return warp.jax_experimental.jax_kernel(
            wp.overload(kernel, arg_types_factory), **kwargs
        )
    return _FfiKernel(
        kernel=cast("wp.Kernel", kernel),
        options=kwargs,
        arg_types_factory=arg_types_factory,
    )


jtu.register_static(_WarpFfiKernel)
