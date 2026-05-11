"""Interop helpers between JAX arrays and NVIDIA Warp.

Use [`to_warp`][liblaf.jarp.warp.to_warp] for array conversion,
[`jax_callable`][liblaf.jarp.warp.jax_callable] and
[`jax_kernel`][liblaf.jarp.warp.jax_kernel] to expose Warp functions through
JAX tracing, [`struct`][liblaf.jarp.warp.struct] for dtype-specialized Warp
struct declarations, and [`liblaf.jarp.warp.types`][liblaf.jarp.warp.types]
for dtypes that follow JAX's active precision mode.
"""

from lazy_loader import attach_stub

__getattr__, __dir__, __all__ = attach_stub(__name__, __file__)

del attach_stub
