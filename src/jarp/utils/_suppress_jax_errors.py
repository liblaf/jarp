from __future__ import annotations

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
    """Log and swallow selected JAX tracing and indexing errors.

    This context manager catches
    [`jax.errors.JAXTypeError`][jax.errors.JAXTypeError] and
    [`jax.errors.JAXIndexError`][jax.errors.JAXIndexError], logs them with
    [`logging.Logger.exception`][logging.Logger.exception], and suppresses the
    exception. Any other exception type is re-raised unchanged.

    Args:
        msg: Message passed to the logger when a supported JAX error is caught.
        *args: Positional arguments forwarded to
            [`logging.Logger.exception`][logging.Logger.exception].
        logger: Logger to use. Defaults to this module's logger.
        **kwargs: Extra keyword arguments forwarded to
            [`logging.Logger.exception`][logging.Logger.exception].

    Yields:
        ``None`` while the managed block runs.
    """
    try:
        yield None
    except (jax.errors.JAXTypeError, jax.errors.JAXIndexError):
        if logger is None:
            logger: logging.Logger = logging.getLogger(__name__)
        logger.exception(msg, *args, **kwargs)
