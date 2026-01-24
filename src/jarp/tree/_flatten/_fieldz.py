from collections.abc import Callable, Sequence

import fieldz
import jax.tree_util as jtu

from ._codegen import codegen


def register_fieldz[T: type](
    cls: T,
    data_fields: Sequence[str] | None = None,
    meta_fields: Sequence[str] | None = None,
) -> T:
    if data_fields is None:
        data_fields = _filter_fieldz(cls, static=False)
    if meta_fields is None:
        meta_fields = _filter_fieldz(cls, static=True)
    flatten: Callable
    flatten_with_keys: Callable
    unflatten: Callable
    flatten, flatten_with_keys, unflatten = codegen(cls, data_fields, meta_fields)
    jtu.register_pytree_with_keys(cls, flatten_with_keys, unflatten, flatten)
    return cls


def _filter_fieldz(cls: type, *, static: bool) -> list[str]:
    return [
        f.name for f in fieldz.fields(cls) if f.metadata.get("static", False) == static
    ]
