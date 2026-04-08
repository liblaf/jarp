import contextlib
import logging
from collections.abc import Generator, Mapping
from typing import TypedDict, Unpack

import jax


class LogOptions(TypedDict, total=False):
    exc_info: logging._ExcInfoType
    stack_info: bool
    stacklevel: int
    extra: Mapping[str, object]


@contextlib.contextmanager
def suppress_jax_errors(
    msg: object = "",
    *args: object,
    logger: logging.Logger | None = None,
    **kwargs: Unpack[LogOptions],
) -> Generator[None]:
    try:
        yield None
    except (jax.errors.JAXTypeError, jax.errors.JAXIndexError):
        if logger is None:
            logger: logging.Logger = logging.getLogger(__name__)
        logger.exception(msg, *args, **kwargs)
