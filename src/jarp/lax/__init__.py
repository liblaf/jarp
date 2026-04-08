"""Control-flow wrappers with automatic Python fallbacks.

Use [jarp.lax][] when you want to try the corresponding `jax.lax` primitive
first, but still rerun the same callback structure eagerly if JAX raises one
of the tracing or indexing errors that commonly appear with Python-only code.
"""

from lazy_loader import attach_stub

__getattr__, __dir__, __all__ = attach_stub(__name__, __file__)

del attach_stub
