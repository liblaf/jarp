from typing import Any

import numpy as np
from jax import Array
from jax._src.literals import TypedNdArray

_ARRAY_TYPES: tuple[type, ...] = (Array, TypedNdArray, np.ndarray, np.generic)


def is_array(element: Any) -> bool:
    # ref: <https://github.com/patrick-kidger/equinox/blob/d1718838dcfff2b0adc4f7a795f72da7bdbec1aa/equinox/_filters.py#L44-L46>
    return isinstance(element, _ARRAY_TYPES)
