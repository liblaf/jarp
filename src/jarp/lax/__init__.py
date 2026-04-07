"""Control-flow wrappers with optional eager fallbacks.

Use [jarp.lax][] when you want code that usually delegates to
`jax.lax` but can still run the same callback structure eagerly during
debugging, tests, or non-jitted execution.
"""

from lazy_loader import attach_stub

__getattr__, __dir__, __all__ = attach_stub(__name__, __file__)

del attach_stub
