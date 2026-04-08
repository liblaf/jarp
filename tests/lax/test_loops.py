from jarp import lax


def test_fori_loop_falls_back_for_python_only_bodies() -> None:
    def body(i: int, total: int) -> int:
        return total + [10, 20, 30][i]

    assert lax.fori_loop(0, 3, body, 0, unroll=2) == 60


def test_while_loop_falls_back_for_python_only_bodies() -> None:
    def cond_fun(state: tuple[int, int]) -> bool:
        i, _ = state
        return i < 3

    def body_fun(state: tuple[int, int]) -> tuple[int, int]:
        i, total = state
        return i + 1, total + [10, 20, 30][i]

    assert lax.while_loop(cond_fun, body_fun, (0, 0)) == (3, 60)
