import ast
import linecache
import types
from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

import jax.tree_util as jtu

from jarp.tree._filters import is_data

from ._codegen import codegen_flatten, codegen_flatten_with_keys, codegen_unflatten

type _AuxData = tuple[Any, ...]
type _Children = tuple[Any, ...]
type _ChildrenWithKeys = tuple[tuple[_KeyEntry, Any], ...]
type _KeyEntry = Any


class PyTreeFunctions[T](NamedTuple):
    flatten: Callable[[T], tuple[_Children, _AuxData]]
    unflatten: Callable[[_AuxData, _Children], T]
    flatten_with_keys: Callable[[T], tuple[_ChildrenWithKeys, _AuxData]]


def codegen_pytree_functions(
    cls: type,
    data_fields: Sequence[str] = (),
    meta_fields: Sequence[str] = (),
    auto_fields: Sequence[str] = (),
    *,
    filter_spec: Callable[[Any], bool] = is_data,
    bypass_setattr: bool | None = None,
) -> PyTreeFunctions:
    if bypass_setattr is None:
        bypass_setattr = cls.__setattr__ is not object.__setattr__
    flatten_def: ast.FunctionDef = codegen_flatten(
        data_fields, meta_fields, auto_fields
    )
    flatten_with_keys_def: ast.FunctionDef = codegen_flatten_with_keys(
        data_fields, meta_fields, auto_fields
    )
    unflatten_def: ast.FunctionDef = codegen_unflatten(
        data_fields, meta_fields, auto_fields, bypass_setattr=bypass_setattr
    )
    module: ast.Module = ast.Module(
        body=[flatten_def, flatten_with_keys_def, unflatten_def], type_ignores=[]
    )
    module = ast.fix_missing_locations(module)
    source: str = ast.unparse(module)
    namespace: dict = {
        "_cls": cls,
        "_filter_spec": filter_spec,
        "_object_new": object.__new__,
        "_object_setattr": object.__setattr__,
        **_make_keys((*data_fields, *meta_fields, *auto_fields)),
    }
    filename: str = _make_filename(cls)
    # use unparse source so we have correct source code locations
    code: types.CodeType = compile(source, filename, "exec")
    exec(code, namespace)  # noqa: S102
    _update_linecache(source, filename)
    return PyTreeFunctions(
        _add_dunder(cls, namespace["flatten"]),
        _add_dunder(cls, namespace["unflatten"]),
        _add_dunder(cls, namespace["flatten_with_keys"]),
    )


def register_generic(
    cls: type,
    data_fields: Sequence[str] = (),
    meta_fields: Sequence[str] = (),
    auto_fields: Sequence[str] = (),
    *,
    filter_spec: Callable[[Any], bool] = is_data,
    bypass_setattr: bool | None = None,
) -> None:
    flatten: Callable
    unflatten: Callable
    flatten_with_keys: Callable
    flatten, unflatten, flatten_with_keys = codegen_pytree_functions(
        cls,
        data_fields,
        meta_fields,
        auto_fields,
        filter_spec=filter_spec,
        bypass_setattr=bypass_setattr,
    )
    jtu.register_pytree_node(cls, flatten, unflatten, flatten_with_keys)


def _add_dunder[C: Callable](cls: type, func: C) -> C:
    func.__module__ = cls.__module__
    return func


def _make_filename(cls: type) -> str:
    return f"<jarp generated functions {cls.__module__}.{cls.__qualname__}>"


def _make_keys(fields: Sequence[str]) -> dict[str, Any]:
    return {f"_{name}_key": jtu.GetAttrKey(name) for name in fields}


def _update_linecache(source: str, filename: str) -> None:
    linecache.cache[filename] = (
        len(source),  # size
        None,  # mtime
        source.splitlines(keepends=True),  # lines
        filename,  # fullname
    )
