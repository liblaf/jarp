import os

import jax

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_enable_x64", True)  # noqa: FBT003
jax.config.update("jax_check_tracer_leaks", True)  # noqa: FBT003
jax.config.update("jax_debug_nans", True)  # noqa: FBT003
