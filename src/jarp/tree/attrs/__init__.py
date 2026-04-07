"""`attrs` helpers for classes that should behave like JAX PyTrees.

The decorators and field specifiers in [jarp.tree.attrs][] wrap `attrs` while
recording which fields should flatten as dynamic data,
remain static metadata, or be decided from the runtime value.
"""

from lazy_loader import attach_stub

__getattr__, __dir__, __all__ = attach_stub(__name__, __file__)

del attach_stub
