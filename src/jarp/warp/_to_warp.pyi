from typing import Any

import jax
import numpy as np
import warp as wp

def to_warp(
    arr: wp.array | np.ndarray | jax.Array,
    dtype: tuple[int, Any] | tuple[int, int, Any] | Any = ...,
    *,
    requires_grad: bool = ...,
    **kwargs,
) -> wp.array: ...
