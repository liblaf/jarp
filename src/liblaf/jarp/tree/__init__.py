"""Helpers for defining, flattening, and transforming JAX PyTrees.

Most users start with [`define`][liblaf.jarp.tree.define],
[`frozen`][liblaf.jarp.tree.frozen], field specifiers such as
[`array`][liblaf.jarp.tree.array] and [`static`][liblaf.jarp.tree.static], and
[`ravel`][liblaf.jarp.tree.ravel]. Lower-level partitioning, registration, and
code-generation helpers remain available for custom integrations. Importing
this package also registers JAX adapters for bound methods and `warp.array`.
"""

from lazy_loader import attach_stub

from .prelude import _prelude

__getattr__, __dir__, __all__ = attach_stub(__name__, __file__)

del attach_stub, _prelude
