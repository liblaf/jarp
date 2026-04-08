"""Utilities for mixed JAX PyTrees and Warp interop.

The top-level package re-exports the filtered call wrappers
[`filter_jit`][jarp.filter_jit] and [`fallback_jit`][jarp.fallback_jit], the
most common helpers from [jarp.tree][], and Warp integration utilities from
[jarp.warp][]. Import [jarp.lax][], [jarp.tree][], or [jarp.warp][] directly
when you need the larger submodule surfaces.
"""

from lazy_loader import attach_stub

__getattr__, __dir__, __all__ = attach_stub(__name__, __file__)

del attach_stub
