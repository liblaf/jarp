"""Helpers for defining, flattening, and transforming JAX PyTrees.

Most users start with [`define`][jarp.tree.define],
[`frozen`][jarp.tree.frozen], field specifiers such as
[`array`][jarp.tree.array] and [`static`][jarp.tree.static], and
[`ravel`][jarp.tree.ravel]. Lower-level partitioning, registration, and
code-generation helpers remain available for custom integrations.
"""

from lazy_loader import attach_stub

__getattr__, __dir__, __all__ = attach_stub(__name__, __file__)

del attach_stub
