from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike, Bool


def select[T](
    condlist: Sequence[Bool[ArrayLike, " ..."]], choicelist: Sequence[T], default: T
) -> T:
    """Select among matching PyTrees with `jax.numpy.select`.

    Each leaf is selected independently with ordered conditions. The first true
    condition at each position selects the corresponding choice leaf; `default`
    supplies the leaf where no condition is true.

    Args:
        condlist: Non-empty sequence of boolean scalar or array-like
            conditions.
        choicelist: PyTrees to choose from. It must have the same length as
            `condlist`, and every choice must have the same tree structure as
            `default`.
        default: PyTree returned where no condition is true.

    Returns:
        A PyTree with the same structure as `default`.

    Raises:
        ValueError: If `condlist` is empty or its length does not match
            `choicelist`.

    Examples:
        >>> import jax.numpy as jnp
        >>> from liblaf import jarp
        >>> result = jarp.tree.select(
        ...     [jnp.array([False, True, False]), jnp.array([True, True, False])],
        ...     [{"value": jnp.array([1, 1, 1])}, {"value": jnp.array([2, 2, 2])}],
        ...     {"value": jnp.array([9, 9, 9])},
        ... )
        >>> result["value"].tolist()
        [2, 1, 9]
    """
    return jax.tree.map(
        lambda *args: jnp.select(condlist, args[:-1], args[-1]), *choicelist, default
    )


def where[T](condition: Bool[ArrayLike, " ..."], x: T, y: T) -> T:
    """Choose between matching PyTrees with `jax.numpy.where`.

    Args:
        condition: Boolean scalar or array-like condition.
        x: PyTree used where `condition` is true.
        y: PyTree used where `condition` is false.

    Returns:
        A PyTree with the same structure as `x` and `y`.

    Examples:
        >>> import jax.numpy as jnp
        >>> from liblaf import jarp
        >>> result = jarp.tree.where(
        ...     jnp.array([True, False]),
        ...     {"value": jnp.array([1, 2])},
        ...     {"value": jnp.array([3, 4])},
        ... )
        >>> result["value"].tolist()
        [1, 4]
    """
    return jax.tree.map(lambda a, b: jnp.where(condition, a, b), x, y)
