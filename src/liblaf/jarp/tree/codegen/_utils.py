from ast import (
    Assign,
    Attribute,
    Call,
    Constant,
    Expr,
    FunctionDef,
    IfExp,
    Load,
    Name,
    Store,
    Tuple,
    arg,
    arguments,
    expr,
    stmt,
)
from collections.abc import Sequence


def codegen_aux(meta_fields: Sequence[str], auto_fields: Sequence[str]) -> list[expr]:
    return _codegen_children(meta_fields, auto_fields, suffix="_meta")


def codegen_children(
    data_fields: Sequence[str], auto_fields: Sequence[str]
) -> list[expr]:
    return _codegen_children(data_fields, auto_fields, suffix="_data")


def codegen_function_def(name: str, args: list[arg], body: list[stmt]) -> FunctionDef:
    # def name(args...): body...
    return FunctionDef(
        name=name,
        args=arguments(
            posonlyargs=[],
            args=args,
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        ),
        body=body,
        decorator_list=[],
        returns=None,
        type_comment=None,
        type_params=[],
    )


def codegen_object_setattr(name: str, value: expr) -> Expr:
    # _object_setattr(obj, "name", value)
    return Expr(
        Call(
            Name("_object_setattr", Load()),
            [Name("obj", Load()), Constant(name), value],
            [],
        )
    )


def codegen_partition(auto_fields: Sequence[str]) -> list[stmt]:
    """.

    ```python
    _a_auto = obj.a
    _b_auto = obj.b
    ...
    _a_data, _a_meta = (_a_auto, None) if _filter_spec(_a_auto) else (None, _a_auto)
    _b_data, _b_meta = (_b_auto, None) if _filter_spec(_b_auto) else (None, _b_auto)
    ...
    ```
    """
    body: list[stmt] = [
        # _name_auto = obj.name
        Assign(
            [Name(f"_{name}_auto", Store())],
            Attribute(Name("obj", Load()), name, Load()),
        )
        for name in auto_fields
    ]
    body.extend(
        # _name_data, _name_meta = (_name_auto, None) if _filter_spec(_name_auto) else (None, _name_auto)
        Assign(
            [
                Tuple(
                    [Name(f"_{name}_data", Store()), Name(f"_{name}_meta", Store())],
                    Store(),
                )
            ],
            IfExp(
                Call(Name("_filter_spec", Load()), [Name(f"_{name}_auto", Load())], []),
                Tuple([Name(f"_{name}_auto", Load()), Constant(None)], Load()),
                Tuple([Constant(None), Name(f"_{name}_auto", Load())], Load()),
            ),
        )
        for name in auto_fields
    )
    return body


def codegen_unpack(attrs: Sequence[str], names: Sequence[str], value: expr) -> Assign:
    # (obj.attr..., name...) = value
    targets: list[expr] = [
        Attribute(Name("obj", Load()), attr, Store()) for attr in attrs
    ]
    targets.extend(Name(name, Store()) for name in names)
    return Assign([Tuple(targets, Store())], value)


def _codegen_children(
    fields: Sequence[str], auto_fields: Sequence[str], suffix: str
) -> list[expr]:
    children: list[expr] = []
    children.extend(Attribute(Name("obj", Load()), name, Load()) for name in fields)
    children.extend(Name(f"_{name}{suffix}", Load()) for name in auto_fields)
    return children
