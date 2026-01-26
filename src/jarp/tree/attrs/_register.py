from collections.abc import Callable, Sequence
from typing import Any

import fieldz

from jarp.tree._filters import is_data
from jarp.tree.codegen import register_generic


def register_fieldz[T: type](
    cls: T,
    data_fields: Sequence[str] | None = None,
    meta_fields: Sequence[str] | None = None,
    auto_fields: Sequence[str] | None = None,
    *,
    filter_spec: Callable[[Any], bool] = is_data,
    bypass_setattr: bool | None = None,
) -> T:
    if data_fields is None:
        data_fields = _filter_field_names(_filter_data, cls)
    if meta_fields is None:
        meta_fields = _filter_field_names(_filter_meta, cls)
    if auto_fields is None:
        auto_fields = _filter_field_names(_filter_auto, cls)
    register_generic(
        cls,
        data_fields,
        meta_fields,
        auto_fields,
        filter_spec=filter_spec,
        bypass_setattr=bypass_setattr,
    )
    return cls


def _filter_field_names(
    function: Callable[[fieldz.Field], bool], cls: type
) -> list[str]:
    return [f.name for f in fieldz.fields(cls) if function(f)]


def _filter_data(field: fieldz.Field) -> bool:
    return not (_filter_auto(field) or _filter_meta(field))


def _filter_meta(field: fieldz.Field) -> bool:
    return field.metadata.get("static", False)


def _filter_auto(field: fieldz.Field) -> bool:
    return field.metadata.get("auto", False)
