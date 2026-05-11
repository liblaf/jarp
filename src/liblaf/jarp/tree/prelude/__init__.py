"""PyTree-aware wrappers for callables and transparent object proxies.

This subpackage contains helper wrappers such as
[`Partial`][liblaf.jarp.tree.Partial] and
[`PyTreeProxy`][liblaf.jarp.tree.PyTreeProxy]. Importing
[`liblaf.jarp.tree`][liblaf.jarp.tree] also imports this package's private
prelude module, which registers bound methods and `warp.array` with JAX before
the public tree helpers are used.
"""

from lazy_loader import attach_stub

__getattr__, __dir__, __all__ = attach_stub(__name__, __file__)

del attach_stub
