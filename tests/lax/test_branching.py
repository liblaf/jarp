import jax.numpy as jnp

from jarp import lax


def test_cond_falls_back_for_python_only_branches() -> None:
    def true_fun(i: int) -> int:
        return [10, 20][i]

    def false_fun(i: int) -> int:
        return -1

    assert lax.cond(True, true_fun, false_fun, 1) == 20
    assert lax.cond(False, true_fun, false_fun, 1) == -1


def test_switch_clamps_indices_for_python_fallbacks() -> None:
    branches = [
        lambda i: [10, 20][i],
        lambda i: [30, 40][i],
    ]
    assert lax.switch(-5, branches, 1) == 20
    assert lax.switch(99, branches, 1) == 40


def test_cond_accepts_jax_compatible_operands() -> None:
    result = lax.cond(
        jnp.array(True),
        lambda x: x + 1,
        lambda x: x - 1,
        jnp.array(3),
    )
    assert int(result) == 4
