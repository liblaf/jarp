from ast import (
    Assign,
    Attribute,
    Call,
    Compare,
    Constant,
    FunctionDef,
    IfExp,
    Is,
    Load,
    Name,
    Return,
    Store,
    Tuple,
    arg,
    expr,
    stmt,
)
from collections.abc import Sequence

from ._utils import (
    codegen_aux,
    codegen_children,
    codegen_function_def,
    codegen_object_setattr,
    codegen_partition,
    codegen_unpack,
)


def codegen_flatten(
    data_fields: Sequence[str], meta_fields: Sequence[str], auto_fields: Sequence[str]
) -> FunctionDef:
    body: list[stmt] = codegen_partition(auto_fields)
    children: list[expr] = codegen_children(data_fields, auto_fields)
    aux: list[expr] = codegen_aux(meta_fields, auto_fields)
    body.append(Return(Tuple([Tuple(children, Load()), Tuple(aux, Load())], Load())))
    return codegen_function_def("flatten", [arg("obj")], body)


def codegen_flatten_with_keys(
    data_fields: Sequence[str], meta_fields: Sequence[str], auto_fields: Sequence[str]
) -> FunctionDef:
    body: list[stmt] = codegen_partition(auto_fields)
    children: list[expr] = codegen_children(data_fields, auto_fields)
    aux: list[expr] = codegen_aux(meta_fields, auto_fields)
    keys: list[expr] = [
        Name(f"_{name}_key", Load()) for name in (*data_fields, *auto_fields)
    ]
    children_with_keys: list[expr] = [
        Tuple([key, child], Load()) for key, child in zip(keys, children, strict=True)
    ]
    body.append(
        Return(Tuple([Tuple(children_with_keys, Load()), Tuple(aux, Load())], Load()))
    )
    return codegen_function_def("flatten_with_keys", [arg("obj")], body)


def codegen_unflatten(
    data_fields: Sequence[str],
    meta_fields: Sequence[str],
    auto_fields: Sequence[str],
    *,
    bypass_setattr: bool = False,
) -> FunctionDef:
    body: list[stmt] = [
        Assign(
            [Name("obj", Store())],
            Call(Name("_object_new", Load()), [Name("_cls", Load())], []),
        )
    ]
    if bypass_setattr:
        body.extend(_codegen_unflatten_bypass(data_fields, meta_fields, auto_fields))
    else:
        body.extend(_codegen_unflatten_direct(data_fields, meta_fields, auto_fields))
    body.append(Return(Name("obj", Load())))
    return codegen_function_def("unflatten", [arg("aux"), arg("children")], body)


def _codegen_unflatten_bypass(
    data_fields: Sequence[str], meta_fields: Sequence[str], auto_fields: Sequence[str]
) -> list[stmt]:
    body: list[stmt] = [
        # _c_meta, _d_meta, ..., _e_meta, _f_meta = aux
        codegen_unpack(
            (),
            [f"_{name}_meta" for name in (*meta_fields, *auto_fields)],
            Name("aux", Load()),
        ),
        # _a_data, _b_data, ..., _e_data, _f_data = children
        codegen_unpack(
            (),
            [f"_{name}_data" for name in (*data_fields, *auto_fields)],
            Name("children", Load()),
        ),
    ]
    body.extend(
        # _object_setattr(obj, "name", _name_data)
        codegen_object_setattr(name, Name(f"_{name}_data", Load()))
        for name in data_fields
    )
    body.extend(
        # _object_setattr(obj, "name", _name_meta)
        codegen_object_setattr(name, Name(f"_{name}_meta", Load()))
        for name in meta_fields
    )
    body.extend(
        # _object_setattr(obj, "name", _name_data if _name_meta is None else _name_meta)
        codegen_object_setattr(
            name,
            # _name_data if _name_meta is None else _name_meta
            IfExp(
                # _name_meta is None
                Compare(Name(f"_{name}_meta", Load()), [Is()], [Constant(None)]),
                Name(f"_{name}_data", Load()),
                Name(f"_{name}_meta", Load()),
            ),
        )
        for name in auto_fields
    )
    return body


def _codegen_unflatten_direct(
    data_fields: Sequence[str], meta_fields: Sequence[str], auto_fields: Sequence[str]
) -> list[stmt]:
    body: list[stmt] = [
        # obj.c, obj.d, ..., _e_meta, _f_meta, ... = aux
        codegen_unpack(
            meta_fields, [f"_{name}_meta" for name in auto_fields], Name("aux", Load())
        ),
        # obj.a, obj.b, ..., _e_data, _f_data, ... = children
        codegen_unpack(
            data_fields,
            [f"_{name}_data" for name in auto_fields],
            Name("children", Load()),
        ),
    ]
    body.extend(
        # obj.name = _name_data if _name_meta is None else _name_meta
        Assign(
            [Attribute(Name("obj", Load()), name, Store())],
            # _name_data if _name_meta is None else _name_meta
            IfExp(
                # _name_meta is None
                Compare(Name(f"_{name}_meta", Load()), [Is()], [Constant(None)]),
                Name(f"_{name}_data", Load()),
                Name(f"_{name}_meta", Load()),
            ),
        )
        for name in auto_fields
    )
    return body
