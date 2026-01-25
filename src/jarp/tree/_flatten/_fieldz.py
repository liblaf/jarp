from collections.abc import Sequence

import fieldz
import jax.tree_util as jtu

from ._codegen import codegen
from ._types import FlattenFunction, FlattenWithKeysFunction, UnflattenFunction


def register_fieldz[T: type](
    cls: T,
    data_fields: Sequence[str] | None = None,
    meta_fields: Sequence[str] | None = None,
) -> T:
    if data_fields is None:
        data_fields = _filter_field_names(cls, static=False)
    if meta_fields is None:
        meta_fields = _filter_field_names(cls, static=True)
    flatten: FlattenFunction
    flatten_with_keys: FlattenWithKeysFunction
    unflatten: UnflattenFunction
    flatten, flatten_with_keys, unflatten = codegen(cls, data_fields, meta_fields)
    jtu.register_pytree_node(cls, flatten, unflatten, flatten_with_keys)
    return cls


def _filter_field_names(cls: type, *, static: bool) -> list[str]:
    return [
        f.name for f in fieldz.fields(cls) if f.metadata.get("static", False) == static
    ]
