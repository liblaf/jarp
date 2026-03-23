import functools
from collections.abc import Callable
from typing import Any, cast


def _wraps[F: Callable](wrapped: F) -> Callable[[Any], F]:
    return cast(
        "F", functools.wraps(wrapped, assigned=("__annotations__", "__type_params__"))
    )
