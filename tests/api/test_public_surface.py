import importlib

import pytest

import liblaf.jarp._jit as jit_module
from liblaf import jarp


def test_top_level_exports_cover_the_runtime_public_surface() -> None:
    assert {
        "Partial",
        "PyTreeProxy",
        "Structure",
        "cond",
        "fallback_jit",
        "filter_jit",
        "first_true_index",
        "fori_loop",
        "lax",
        "ravel",
        "switch",
        "struct",
        "to_warp",
        "tree",
        "warp",
        "while_loop",
    } <= set(jarp.__all__)

    assert "register_pytree_prelude" not in jarp.__all__
    assert "utils" not in jarp.__all__
    assert callable(jarp.filter_jit)
    assert callable(jarp.fallback_jit)
    assert callable(jarp.first_true_index)
    assert callable(jarp.ravel)
    assert callable(jarp.to_warp)


def test_submodule_exports_are_discoverable() -> None:
    assert set(jarp.lax.__all__) == {
        "LaxWrapper",
        "cond",
        "first_true_index",
        "fori_loop",
        "lax_wrapper",
        "switch",
        "while_loop",
    }
    assert set(jit_module.__all__) == {"fallback_jit", "filter_jit"}
    assert {"register_fieldz", "register_generic"} <= set(jarp.tree.__all__)
    assert "register_pytree_prelude" not in jarp.tree.__all__
    assert {"jax_callable", "jax_kernel", "struct", "to_warp", "types"} <= set(
        jarp.warp.__all__
    )


def test_removed_utils_module_is_not_importable() -> None:
    with pytest.raises(ModuleNotFoundError) as exc_info:
        importlib.import_module("liblaf.jarp.utils")

    assert exc_info.value.name == "liblaf.jarp.utils"
