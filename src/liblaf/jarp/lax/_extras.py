from collections.abc import Sequence

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Integer


def first_true_index(condlist: Sequence[ArrayLike]) -> Integer[Array, "*shape"]:
    """Return the index of the first true condition.

    This is a small [`jax.numpy.select`][jax.numpy.select] wrapper for cases
    where an ordered condition list should become integer labels. Each result
    value is the zero-based index of the first true condition at that position.
    When no condition is true, the result is `len(condlist)`.

    Args:
        condlist: Non-empty ordered sequence of scalar or array-like boolean
            conditions. Array conditions follow `jax.numpy.select` broadcasting
            rules.

    Returns:
        A JAX integer array with the broadcast condition shape. Scalar
        conditions return a zero-dimensional array.

    Raises:
        ValueError: If `condlist` is empty.

    Examples:
        >>> from liblaf.jarp.lax import first_true_index
        >>> int(first_true_index([False, True, True]))
        1
        >>> int(first_true_index([False, False]))
        2

        Array conditions are evaluated elementwise.

        >>> import jax.numpy as jnp
        >>> first_true_index(
        ...     [
        ...         jnp.array([False, True, False, False]),
        ...         jnp.array([True, True, False, False]),
        ...         jnp.array([True, False, True, False]),
        ...     ]
        ... ).tolist()
        [1, 0, 2, 3]
    """
    return jnp.select(condlist, range(len(condlist)), default=len(condlist))
