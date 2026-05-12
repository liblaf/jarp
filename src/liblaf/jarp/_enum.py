from __future__ import annotations

import enum
from collections.abc import Sequence
from typing import Self, cast

import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, ArrayLike, Bool, Integer

from liblaf.jarp import tree


class Enum(enum.Enum):
    """JAX-compatible enum base class with traceable integer values.

    `Enum` behaves like [`enum.Enum`][enum.Enum], but its `value` is a JAX
    array leaf. Subclasses are registered as PyTrees, so enum state can travel
    through `jax.jit` and `jax.lax` loops as dynamic data.

    Array-valued results can represent several members at once. In that case
    the enum object's `name` is `"<unknown>"`, while `value` remains the
    traceable integer array that JAX operates on.

    Examples:
        >>> import enum
        >>> import jax.numpy as jnp
        >>> from liblaf import jarp
        >>> class Phase(jarp.Enum):
        ...     START = enum.auto()
        ...     RUNNING = enum.auto()
        ...     DONE = enum.auto()
        >>> int(Phase.RUNNING.value)
        1
        >>> Phase.where(
        ...     jnp.array([True, False]), Phase.START, Phase.RUNNING
        ... ).value.tolist()
        [0, 1]
    """

    def __init_subclass__(cls, **kwargs) -> None:
        """Register each subclass as a keyed JAX PyTree."""
        super().__init_subclass__(**kwargs)
        jtu.register_pytree_with_keys_class(cls)

    def __hash__(self) -> int:
        """Return a hash that keeps enum classes distinct."""
        return hash((type(self), int(self._value_)))

    def __eq__(self, other: object) -> bool:
        """Compare members from the same enum class by value."""
        if not isinstance(other, type(self)):
            return NotImplemented
        return cast("bool", self.value == other.value)

    @staticmethod
    def _generate_next_value_(
        name: str, start: int, count: int, last_values: list[int]
    ) -> int:
        del name, start, last_values
        return count

    @classmethod
    def _missing_(cls, value: object) -> Self:
        """Build a traced enum object from a dynamic integer value."""
        value: Integer[Array, ""] = cast('Integer[Array, ""]', value)
        obj: Self = object.__new__(cls)
        try:
            obj._name_ = cls._value2member_map_[int(value)]._name_
        except Exception:  # noqa: BLE001
            obj._name_ = "<unknown>"
        # Do not validate `value` here; callers are responsible for passing a
        # valid member value. Assigning `value` directly to `obj._value_` keeps
        # JAX tracing working.
        obj._value_ = value
        return obj

    def tree_flatten(self) -> tuple[tuple[Integer[Array, ""]], None]:
        """Flatten the enum value as the only dynamic child."""
        child: Integer[Array, ""] = jnp.asarray(self.value, jnp.int32)
        return (child,), None

    def tree_flatten_with_keys(
        self,
    ) -> tuple[tuple[tuple[jtu.GetAttrKey, Integer[Array, ""]]], None]:
        """Flatten the enum value with a stable `value` path key."""
        key: jtu.GetAttrKey = jtu.GetAttrKey("value")
        child: Integer[Array, ""] = jnp.asarray(self.value, jnp.int32)
        return ((key, child),), None

    @classmethod
    def tree_unflatten(cls, meta: None, data: tuple[Integer[Array, ""]]) -> Self:
        """Rebuild an enum object from its flattened integer value."""
        del meta
        (value,) = data
        return cls._missing_(value)

    @staticmethod
    def select[T](
        condlist: Sequence[Bool[ArrayLike, " ..."]], choicelist: Sequence[T], default: T
    ) -> T:
        """Select among enum-bearing PyTrees with ordered conditions.

        This delegates to [`tree.select`][liblaf.jarp.tree.select]. Conditions
        follow [`jax.numpy.select`][jax.numpy.select] semantics: the first true
        condition at each position selects the corresponding choice, and
        `default` is used where no condition is true.

        Args:
            condlist: Non-empty sequence of boolean scalar or array-like
                conditions.
            choicelist: PyTrees to choose from. It must have the same length as
                `condlist`, and every choice must have the same tree structure
                as `default`.
            default: PyTree returned where no condition is true.

        Returns:
            A PyTree with the same structure as `default`.

        Raises:
            ValueError: If `condlist` is empty or its length does not match
                `choicelist`.

        Examples:
            >>> import enum
            >>> import jax.numpy as jnp
            >>> from liblaf import jarp
            >>> class Phase(jarp.Enum):
            ...     START = enum.auto()
            ...     RUNNING = enum.auto()
            ...     DONE = enum.auto()
            >>> result = Phase.select(
            ...     [jnp.array([False, True]), jnp.array([True, True])],
            ...     [Phase.START, Phase.RUNNING],
            ...     default=Phase.DONE,
            ... )
            >>> result.value.tolist()
            [1, 0]
        """
        return tree.select(condlist, choicelist, default)

    @staticmethod
    def where[T](condition: Bool[ArrayLike, " ..."], x: T, y: T) -> T:
        """Choose between enum-bearing PyTrees leaf by leaf.

        This delegates to [`tree.where`][liblaf.jarp.tree.where]. It applies
        [`jax.numpy.where`][jax.numpy.where] to each matching pair of leaves in
        `x` and `y`.

        Args:
            condition: Boolean scalar or array-like condition.
            x: PyTree used where `condition` is true.
            y: PyTree used where `condition` is false.

        Returns:
            A PyTree with the same structure as `x` and `y`.

        Examples:
            >>> import enum
            >>> import jax.numpy as jnp
            >>> from liblaf import jarp
            >>> class Phase(jarp.Enum):
            ...     START = enum.auto()
            ...     RUNNING = enum.auto()
            >>> Phase.where(
            ...     jnp.array([True, False]), Phase.START, Phase.RUNNING
            ... ).value.tolist()
            [0, 1]
        """
        return tree.where(condition, x, y)

    @property
    def value(self) -> Integer[Array, ""]:
        """Return the enum's dynamic integer value as an `int32` JAX array."""
        return jnp.asarray(self._value_, jnp.int32)
