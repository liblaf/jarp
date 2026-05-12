import enum
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import pytest
from jax import Array
from jaxtyping import ArrayLike, Bool

from liblaf.jarp import Enum


class Phase(Enum):
    START = enum.auto()
    RUNNING = enum.auto()
    DONE = enum.auto()


def test_auto_members_use_zero_based_integer_values() -> None:
    assert int(Phase.START.value) == 0
    assert int(Phase.RUNNING.value) == 1
    assert int(Phase.DONE.value) == 2


def test_members_compare_within_the_same_enum_class() -> None:
    assert bool(Phase.START == Phase.START)
    assert not bool(Phase.START == Phase.RUNNING)


def test_different_enum_classes_do_not_compare_by_raw_value() -> None:
    class Other(Enum):
        START = enum.auto()

    assert Phase.START != Other.START


def test_enum_member_flattens_as_keyed_data_leaf() -> None:
    leaves, treedef = jax.tree.flatten(Phase.START)
    rebuilt = jax.tree.unflatten(treedef, leaves)

    assert [int(leaf) for leaf in leaves] == [0]
    assert isinstance(rebuilt, Phase)
    assert rebuilt.name == "START"
    assert int(rebuilt.value) == 0

    keyed_leaves, _treedef = jtu.tree_flatten_with_path(Phase.START)
    assert len(keyed_leaves) == 1
    path, leaf = keyed_leaves[0]
    assert path == (jtu.GetAttrKey("value"),)
    assert int(leaf) == 0


def test_where_selects_enum_values_under_jit() -> None:
    @jax.jit
    def choose(mask: Bool[ArrayLike, " ..."]) -> Phase:
        return Phase.where(mask, Phase.START, Phase.RUNNING)

    result = choose(jnp.array([True, False, True]))

    assert isinstance(result, Phase)
    assert result.name == "<unknown>"
    assert result.value.tolist() == [0, 1, 0]


def test_select_uses_first_true_condition_and_default_under_jit() -> None:
    @jax.jit
    def choose(first: Array, second: Array) -> Phase:
        return Phase.select(
            [first, second],
            [Phase.START, Phase.RUNNING],
            default=Phase.DONE,
        )

    result = choose(
        jnp.array([False, True, False]),
        jnp.array([True, True, False]),
    )

    assert isinstance(result, Phase)
    assert result.name == "<unknown>"
    assert result.value.tolist() == [1, 0, 2]


@pytest.mark.parametrize(
    ("condlist", "choicelist"),
    [
        ([], []),
        ([True], [Phase.START, Phase.RUNNING]),
        ([True, False], [Phase.START]),
    ],
)
def test_select_rejects_invalid_condition_choice_pairs(
    condlist: Sequence[bool],
    choicelist: Sequence[Phase],
) -> None:
    with pytest.raises(ValueError, match="condlist"):
        Phase.select(condlist, choicelist, default=Phase.DONE)


def test_while_loop_can_carry_enum_values() -> None:
    def cond(carry: tuple[Array, Phase]) -> Array:
        step, _phase = carry
        return step < 3

    def body(carry: tuple[Array, Phase]) -> tuple[Array, Phase]:
        step, phase = carry
        phase = Phase.where(step == 1, Phase.RUNNING, phase)
        return step + 1, phase

    step, phase = jax.lax.while_loop(cond, body, (jnp.array(0), Phase.START))

    assert int(step) == 3
    assert isinstance(phase, Phase)
    assert phase.name == "RUNNING"
    assert int(phase.value) == 1
