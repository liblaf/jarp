"""Code-generation helpers for high-performance PyTree registrations.

These utilities build specialized flatten and unflatten callbacks for classes
whose field layout is known ahead of time.
"""

from lazy_loader import attach_stub

__getattr__, __dir__, __all__ = attach_stub(__name__, __file__)

del attach_stub
