"""Interop helpers between JAX arrays and NVIDIA Warp.

Use [`to_warp`][jarp.warp.to_warp] for array conversion,
[`jax_callable`][jarp.warp.jax_callable] and
[`jax_kernel`][jarp.warp.jax_kernel] to expose Warp functions through JAX
tracing, and [`jarp.warp.types`][jarp.warp.types] for dtypes that follow JAX's
active precision mode.
"""

from lazy_loader import attach_stub

__getattr__, __dir__, __all__ = attach_stub(__name__, __file__)

del attach_stub
