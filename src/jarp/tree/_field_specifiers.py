from typing import Any

import attrs
from liblaf import grapes


@grapes.wraps(attrs.field)
def field(**kwargs) -> Any:
    static: bool = kwargs.pop("static", False)
    if static:
        kwargs["metadata"] = {"static": static, **kwargs.pop("metadata", {})}
    return attrs.field(**kwargs)


@grapes.wraps(attrs.field)
def static(**kwargs) -> Any:
    kwargs.setdefault("static", True)
    return field(**kwargs)
