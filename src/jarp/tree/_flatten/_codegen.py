import ast
import linecache
import types
from collections.abc import Callable, Iterable
from typing import Any, NamedTuple, Protocol

import jax.tree_util as jtu

type KeyEntry = Any
type Leaf = Any

_PREFIX: str = "_jarp_codegen_"
_CLS: str = f"{_PREFIX}cls"
_OBJECT_NEW: str = f"{_PREFIX}object_new"
_OBJECT_SETATTR: str = f"{_PREFIX}object_setattr"


class FlattenFunction(Protocol):
    def __call__(self, obj: Any) -> tuple[tuple[Leaf, ...], tuple[Any, ...]]: ...


class FlattenWithKeysFunction(Protocol):
    def __call__(
        self, obj: Any
    ) -> tuple[tuple[tuple[KeyEntry, Leaf], ...], tuple[Any, ...]]: ...


class UnflattenFunction(Protocol):
    def __call__(self, aux: tuple[Any, ...], children: tuple[Leaf, ...]) -> Any: ...


class CodeGenResults(NamedTuple):
    tree_flatten: FlattenFunction
    tree_flatten_with_keys: FlattenWithKeysFunction
    tree_unflatten: UnflattenFunction


def codegen(
    cls: type, data_fields: Iterable[str], meta_fields: Iterable[str]
) -> CodeGenResults:
    return CodeGenResults(
        _codegen_flatten(cls, data_fields, meta_fields),
        _codegen_flatten_with_keys(cls, data_fields, meta_fields),
        _codegen_unflatten(cls, data_fields, meta_fields),
    )


def _add_dunder(cls: type, func: Callable) -> None:
    func.__module__ = cls.__module__


def _codegen_flatten(
    cls: type, data_fields: Iterable[str], meta_fields: Iterable[str]
) -> FlattenFunction:
    children: ast.Tuple = _codegen_attrgetter(data_fields)
    aux: ast.Tuple = _codegen_attrgetter(meta_fields)
    source: ast.FunctionDef = ast.FunctionDef(
        "tree_flatten",
        ast.arguments(args=[ast.arg("obj")]),
        [ast.Return(ast.Tuple([children, aux]))],
    )
    filename: str = _make_filename(cls, "tree_flatten")
    namespace: dict[str, Any] = {}
    _compile(source, filename, namespace)
    func: FlattenFunction = namespace["tree_flatten"]
    _add_dunder(cls, func)
    return func


def _codegen_flatten_with_keys(
    cls: type, data_fields: Iterable[str], meta_fields: Iterable[str]
) -> FlattenWithKeysFunction:
    children: ast.Tuple = ast.Tuple(
        [
            ast.Tuple(
                [ast.Name(f"{_PREFIX}{name}_key"), ast.Attribute(ast.Name("obj"), name)]
            )
            for name in data_fields
        ]
    )
    aux: ast.Tuple = _codegen_attrgetter(meta_fields)
    source: ast.FunctionDef = ast.FunctionDef(
        "tree_flatten_with_keys",
        ast.arguments(args=[ast.arg("obj")]),
        [ast.Return(ast.Tuple([children, aux]))],
    )
    filename: str = _make_filename(cls, "tree_flatten_with_keys")
    namespace: dict[str, Any] = {
        f"{_PREFIX}{name}_key": jtu.GetAttrKey(name) for name in data_fields
    }
    _compile(source, filename, namespace)
    func: FlattenWithKeysFunction = namespace["tree_flatten_with_keys"]
    _add_dunder(cls, func)
    return func


def _codegen_unflatten(
    cls: type, data_fields: Iterable[str], meta_fields: Iterable[str]
) -> UnflattenFunction:
    body: list[ast.stmt] = [
        # obj = _object_new(_cls)
        ast.Assign(
            [ast.Name("obj", ast.Store())],
            ast.Call(ast.Name(_OBJECT_NEW), [ast.Name(_CLS)]),
        )
    ]
    if cls.__setattr__ is object.__setattr__:
        # obj.a, obj.b, ... = aux
        body.extend(_codegen_unpack("aux", meta_fields))
        # obj.a, obj.b, ... = children
        body.extend(_codegen_unpack("children", data_fields))
    else:
        # bypass cls.__setattr__
        # _a, _b, ... = aux
        # _object_setattr(obj, "a", _a)
        body.extend(_codegen_setattr("aux", meta_fields))
        # _a, _b, ... = children
        # _object_setattr(obj, "a", _a)
        body.extend(_codegen_setattr("children", data_fields))
    # return obj
    body.append(ast.Return(ast.Name("obj")))
    source: ast.FunctionDef = ast.FunctionDef(
        "tree_unflatten",
        ast.arguments(args=[ast.arg("aux"), ast.arg("children")]),
        body,
    )
    filename: str = _make_filename(cls, "tree_unflatten")
    namespace: dict[str, Any] = {
        _CLS: cls,
        _OBJECT_NEW: object.__new__,
        _OBJECT_SETATTR: object.__setattr__,
    }
    _compile(source, filename, namespace)
    func: UnflattenFunction = namespace["tree_unflatten"]
    _add_dunder(cls, func)
    return func


def _codegen_attrgetter(fields: Iterable[str]) -> ast.Tuple:
    return ast.Tuple([ast.Attribute(ast.Name("obj"), name) for name in fields])


def _codegen_unpack(arg_name: str, fields: Iterable[str]) -> list[ast.stmt]:
    return [
        # obj.a, obj.b, ... = arg
        ast.Assign(
            [
                ast.Tuple(
                    [
                        ast.Attribute(ast.Name("obj"), name, ast.Store())
                        for name in fields
                    ],
                    ast.Store(),
                )
            ],
            ast.Name(arg_name),
        )
    ]


def _codegen_setattr(arg_name: str, fields: Iterable[str]) -> list[ast.stmt]:
    # My benchmarks show that unpacking is slightly faster than indexing,
    # so we use unpacking even for setattr.
    body: list[ast.stmt] = [
        # _a, _b, ... = arg
        ast.Assign(
            [
                ast.Tuple(
                    [ast.Name(f"{_PREFIX}{name}", ast.Store()) for name in fields],
                    ast.Store(),
                )
            ],
            ast.Name(arg_name),
        )
    ]
    body.extend(
        [
            # _object_setattr(obj, "name", _name)
            ast.Expr(
                ast.Call(
                    ast.Name(_OBJECT_SETATTR),
                    [ast.Name("obj"), ast.Constant(name), ast.Name(f"{_PREFIX}{name}")],
                )
            )
            for name in fields
        ]
    )
    return body


def _compile(source: ast.FunctionDef, filename: str, namespace: dict[str, Any]) -> None:
    module: ast.Module = ast.Module([source])
    module: ast.Module = ast.fix_missing_locations(module)
    code: types.CodeType = compile(module, filename, "exec")
    exec(code, namespace)  # noqa: S102
    _update_linecache(module, filename)


def _make_filename(cls: type, func_name: str) -> str:
    return f"<jarp generated {func_name} {cls.__module__}.{cls.__qualname__}>"


def _make_keys(fields: Iterable[str]) -> dict[str, KeyEntry]:
    return {f"{_PREFIX}{name}_key": jtu.GetAttrKey(name) for name in fields}


def _update_linecache(source: ast.Module, filename: str) -> None:
    code: str = ast.unparse(source)
    linecache.cache[filename] = (
        len(code),  # size
        None,  # mtime
        code.splitlines(keepends=True),  # lines
        filename,  # fullname
    )
