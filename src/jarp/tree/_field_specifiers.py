from collections.abc import Mapping
from typing import Any

import attrs
import tlz
from liblaf import grapes


@grapes.wraps(attrs.field)
def field(**kwargs) -> Any:
    static: bool = kwargs.pop("static", False)
    if static:
        metadata: Mapping = kwargs.get("metadata", {})
        metadata = tlz.assoc(metadata, "static", static)
        kwargs["metadata"] = metadata
    return attrs.field(**kwargs)


@grapes.wraps(attrs.field)
def static(**kwargs) -> Any:
    kwargs.setdefault("static", True)
    return field(**kwargs)
