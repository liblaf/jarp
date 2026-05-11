import jax.numpy as jnp

from liblaf.jarp.lax import first_true_index


def test_first_true_index_returns_earliest_true_condition() -> None:
    assert int(first_true_index([False, True, True])) == 1


def test_first_true_index_returns_condition_count_when_none_are_true() -> None:
    assert int(first_true_index([False, False])) == 2


def test_first_true_index_vectorizes_over_array_conditions() -> None:
    result = first_true_index(
        [
            jnp.array([False, True, False, False]),
            jnp.array([True, True, False, False]),
            jnp.array([True, False, True, False]),
        ]
    )

    assert result.tolist() == [1, 0, 2, 3]
