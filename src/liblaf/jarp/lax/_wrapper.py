import functools
import logging
from collections.abc import Callable
from typing import Any

import jax

from liblaf.jarp import tree

logger: logging.Logger = logging.getLogger(__name__)


@tree.frozen_static(slots=False)
class LaxWrapper[**P, T]:
    """Call a JAX primitive first and cache Python fallback signatures.

    `LaxWrapper` powers the public helpers in
    [`liblaf.jarp.lax`][liblaf.jarp.lax]. It preserves wrapper metadata from
    the wrapped JAX primitive when that metadata exists, tries that primitive
    on each new call shape, and records metadata signatures that should skip
    directly to the Python fallback after a supported JAX error. Callable
    objects without ordinary function metadata are accepted.

    Examples:
        >>> from liblaf.jarp.lax import LaxWrapper
        >>> class Wrapped:
        ...     def __call__(self, value):
        ...         return value + 1
        >>> wrapper = LaxWrapper(Wrapped(), lambda value: value - 1)
        >>> wrapper(2)
        3

    Attributes:
        __wrapped__: JAX callable attempted before the fallback.
        fallback: Python callable used after selected JAX tracing or indexing
            errors.
        success_cache: Mapping from partitioned input metadata to whether the
            JAX path is known to work. Failed signatures are stored as `False`.
    """

    __wrapped__: Callable[P, T] = tree.static(alias="__wrapped__")
    fallback: Callable[P, T] = tree.static()
    success_cache: dict[tree.AuxData, bool] = tree.field(factory=dict)

    def __attrs_post_init__(self) -> None:
        for attr in functools.WRAPPER_ASSIGNMENTS:
            try:
                value = getattr(self.__wrapped__, attr)
            except AttributeError:
                pass
            else:
                object.__setattr__(self, attr, value)
        for attr in functools.WRAPPER_UPDATES:
            getattr(self, attr).update(getattr(self.__wrapped__, attr, {}))

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        __tracebackhide__ = True
        if self.success_cache:
            _inputs_data, inputs_meta = tree.partition((args, kwargs))
            if self.success_cache.get(inputs_meta) is False:
                return self.fallback(*args, **kwargs)
        try:
            return self.__wrapped__(*args, **kwargs)
        except (jax.errors.JAXTypeError, jax.errors.JAXIndexError):
            logger.exception("", stacklevel=2)
        _inputs_data, inputs_meta = tree.partition((args, kwargs))
        self.success_cache[inputs_meta] = False
        return self.fallback(*args, **kwargs)


def lax_wrapper[**P, T](
    wrapped: Callable[..., Any],  # jax's typing is not precise, so we loosen it here
) -> Callable[[Callable[P, T]], LaxWrapper[P, T]]:
    """Decorate an eager fallback with a [`LaxWrapper`][liblaf.jarp.lax.LaxWrapper].

    Args:
        wrapped: JAX primitive or compatible callable to try first.

    Returns:
        A decorator that turns the fallback function into a `LaxWrapper`.
    """
    return functools.partial(LaxWrapper, wrapped)
