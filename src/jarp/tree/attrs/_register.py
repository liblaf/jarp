from collections.abc import Callable, Sequence
from typing import Any

import fieldz

from jarp.tree._filters import is_data
from jarp.tree.codegen import register_generic

from ._field_specifiers import FieldType


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
        data_fields = _filter_field_names(cls, FieldType.DATA)
    if meta_fields is None:
        meta_fields = _filter_field_names(cls, FieldType.META)
    if auto_fields is None:
        auto_fields = _filter_field_names(cls, FieldType.AUTO)
    register_generic(
        cls,
        data_fields,
        meta_fields,
        auto_fields,
        filter_spec=filter_spec,
        bypass_setattr=bypass_setattr,
    )
    return cls


def _filter_field_names(cls: type, field_type: FieldType) -> list[str]:
    return [
        f.name
        for f in fieldz.fields(cls)
        if FieldType(f.metadata.get("static")) is field_type
    ]
