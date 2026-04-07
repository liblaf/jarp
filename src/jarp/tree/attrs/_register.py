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
    """Register an ``attrs`` class with JAX using field metadata.

    Field groups default to the metadata written by
    [`array`][jarp.tree.array], [`auto`][jarp.tree.auto], and
    [`static`][jarp.tree.static]. Pass explicit field lists when you need to
    register a class that was not declared with jarp's field helpers.

    Args:
        cls: Class to register.
        data_fields: Field names that are always treated as dynamic children.
        meta_fields: Field names that are always treated as static metadata.
        auto_fields: Field names filtered at runtime with ``filter_spec``.
        filter_spec: Predicate used to split ``auto_fields`` into dynamic data
            or metadata.
        bypass_setattr: Whether generated unflattening code should use
            [`object.__setattr__`][object.__setattr__] instead of normal
            attribute assignment.

    Returns:
        The same class object, for decorator-style usage.
    """
    if data_fields is None:
        data_fields: list[str] = _filter_field_names(cls, FieldType.DATA)
    if meta_fields is None:
        meta_fields: list[str] = _filter_field_names(cls, FieldType.META)
    if auto_fields is None:
        auto_fields: list[str] = _filter_field_names(cls, FieldType.AUTO)
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
