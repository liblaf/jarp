"""Utilities for mixed JAX PyTrees and NVIDIA Warp interop.

The top-level package re-exports the filtered call wrappers
[`filter_jit`][liblaf.jarp.filter_jit] and
[`fallback_jit`][liblaf.jarp.fallback_jit], selected
[`liblaf.jarp.lax`][liblaf.jarp.lax] helpers, the most common helpers from
[liblaf.jarp.tree][], and Warp integration utilities such as
[`to_warp`][liblaf.jarp.to_warp], [`struct`][liblaf.jarp.struct],
[`jax_callable`][liblaf.jarp.jax_callable], and
[`jax_kernel`][liblaf.jarp.jax_kernel]. Import [liblaf.jarp.lax][],
[liblaf.jarp.tree][], or [liblaf.jarp.warp][] directly when you need the
larger submodule surfaces.
"""

from lazy_loader import attach_stub

__getattr__, __dir__, __all__ = attach_stub(__name__, __file__)

del attach_stub
