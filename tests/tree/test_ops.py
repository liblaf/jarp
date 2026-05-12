import jax
import jax.numpy as jnp
import pytest
from jax import Array

from liblaf.jarp import tree


def test_where_selects_matching_pytree_leaves() -> None:
    x = {"label": jnp.array([10, 20]), "score": jnp.array([1.0, 2.0])}
    y = {"label": jnp.array([30, 40]), "score": jnp.array([3.0, 4.0])}

    result = tree.where(jnp.array([True, False]), x, y)

    assert result["label"].tolist() == [10, 40]
    assert result["score"].tolist() == [1.0, 4.0]


def test_select_uses_first_true_condition_for_each_pytree_leaf() -> None:
    default = {"value": jnp.array([9, 9, 9])}
    choices = [
        {"value": jnp.array([1, 1, 1])},
        {"value": jnp.array([2, 2, 2])},
    ]

    result = tree.select(
        [
            jnp.array([False, True, False]),
            jnp.array([True, True, False]),
        ],
        choices,
        default,
    )

    assert result["value"].tolist() == [2, 1, 9]


def test_select_works_under_jit_with_static_choices() -> None:
    @jax.jit
    def choose(first: Array, second: Array) -> dict[str, Array]:
        return tree.select(
            [first, second],
            [
                {"value": jnp.array([1, 1, 1])},
                {"value": jnp.array([2, 2, 2])},
            ],
            {"value": jnp.array([9, 9, 9])},
        )

    result = choose(
        jnp.array([False, True, False]),
        jnp.array([True, True, False]),
    )

    assert result["value"].tolist() == [2, 1, 9]


@pytest.mark.parametrize(
    ("condlist", "choicelist"),
    [
        ([], []),
        ([True], [{"value": jnp.array(1)}, {"value": jnp.array(2)}]),
        ([True, False], [{"value": jnp.array(1)}]),
    ],
)
def test_select_rejects_invalid_condition_choice_pairs(
    condlist: list[bool],
    choicelist: list[dict[str, Array]],
) -> None:
    with pytest.raises(ValueError, match="condlist"):
        tree.select(condlist, choicelist, {"value": jnp.array(0)})
