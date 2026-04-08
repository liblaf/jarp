import jarp
import jarp._jit as jit_module


def test_public_runtime_exports_are_available() -> None:
    assert callable(jarp.filter_jit)
    assert callable(jarp.fallback_jit)
    assert callable(jarp.lax.cond)
    assert callable(jarp.tree.ravel)
    assert callable(jarp.warp.to_warp)
    assert callable(jarp.utils.suppress_jax_errors)
    assert callable(jit_module.filter_jit)
    assert isinstance(jarp.__version__, str)
