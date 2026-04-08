import jax
import pytest
import warp as wp


def pytest_configure(config: pytest.Config) -> None:
    del config
    jax.config.update("jax_platforms", "cpu")
    jax.config.update("jax_check_tracer_leaks", True)  # noqa: FBT003
    jax.config.update("jax_debug_nans", True)  # noqa: FBT003
    jax.config.update("jax_enable_x64", True)  # noqa: FBT003
    wp.init()
