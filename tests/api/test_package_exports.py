import jarp
import jarp._jit as jit_module


def test_package_exports_cover_the_public_surface() -> None:
    assert {
        "fallback_jit",
        "filter_jit",
        "lax",
        "ravel",
        "to_warp",
        "tree",
        "utils",
        "warp",
        "while_loop",
    } <= set(jarp.__all__)
    assert callable(jarp.filter_jit)
    assert callable(jarp.fallback_jit)
    assert callable(jarp.ravel)
    assert callable(jarp.to_warp)


def test_submodule_exports_remain_discoverable() -> None:
    assert set(jarp.lax.__all__) == {"cond", "fori_loop", "switch", "while_loop"}
    assert set(jit_module.__all__) == {"fallback_jit", "filter_jit"}
    assert {"register_fieldz", "register_generic", "register_pytree_prelude"} <= set(
        jarp.tree.__all__
    )
    assert {"jax_callable", "jax_kernel", "to_warp", "types"} <= set(jarp.warp.__all__)
