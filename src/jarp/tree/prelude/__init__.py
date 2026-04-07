"""PyTree-aware wrappers for callables, proxies, and support registrations.

This subpackage contains helper wrappers such as
[`Partial`][jarp.tree.Partial] and [`PyTreeProxy`][jarp.tree.PyTreeProxy], plus
the one-time registrations used by filtered JIT for bound methods and Warp
arrays.
"""

from lazy_loader import attach_stub

__getattr__, __dir__, __all__ = attach_stub(__name__, __file__)

del attach_stub
