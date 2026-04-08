from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Unpack, overload

import jax

from jarp import tree

from ._filter_jit import FilterJitOptions, Inner, Outer

if TYPE_CHECKING:
    from _typeshed import IdentityFunction


logger: logging.Logger = logging.getLogger(__name__)


@tree.define(slots=False)
class FallbackOuter[**P, T](Outer[P, T]):
    jit_able_cache: dict[tree.AuxData, bool] = tree.field(factory=dict)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        inputs_data, inputs_meta = tree.partition((args, kwargs))
        jit_able: bool | None = self.jit_able_cache.get(inputs_meta)
        if jit_able is False:
            return self._call_no_jit(*args, **kwargs)
        try:
            output_data, output_meta = self.inner(
                inputs_data, self.fun_data, inputs_meta
            )
        except (jax.errors.JAXTypeError, jax.errors.JAXIndexError):
            logger.exception("", stacklevel=2)
        else:
            self.jit_able_cache[inputs_meta] = True
            return tree.combine(output_data, output_meta)
        self.jit_able_cache[inputs_meta] = False
        return self._call_no_jit(*args, **kwargs)

    def reset_cache_entry(self, *args: P.args, **kwargs: P.kwargs) -> None:
        _inputs_data, inputs_meta = tree.partition((args, kwargs))
        self.jit_able_cache.pop(inputs_meta, None)

    def _call_no_jit(self, *args: P.args, **kwargs: P.kwargs) -> T:
        fun: Callable[P, T] = tree.combine(self.fun_data, self.inner.fun_meta)
        return fun(*args, **kwargs)


@overload
def fallback_jit[F: Callable[..., Any]](
    fun: F, **kwargs: Unpack[FilterJitOptions]
) -> F: ...
@overload
def fallback_jit(
    fun: None = None, **kwargs: Unpack[FilterJitOptions]
) -> IdentityFunction: ...
def fallback_jit[**P, T](
    fun: Callable[P, T] | None = None, **kwargs: FilterJitOptions
) -> Callable:
    if fun is None:
        return functools.partial(fallback_jit, **kwargs)
    fun_data, fun_meta = tree.partition(fun)
    inner: Inner = Inner(fun_meta=fun_meta)
    outer: FallbackOuter = FallbackOuter(fun_data=fun_data, inner=inner)
    functools.update_wrapper(outer, fun)
    return outer
