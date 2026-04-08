import jax

import jarp.lax._control as control


def test_fori_loop_falls_back_to_python_when_jax_errors(monkeypatch) -> None:
    def raise_index_error(*args, **kwargs):
        del args, kwargs
        raise jax.errors.JAXIndexError("boom")

    monkeypatch.setattr(control.jax.lax, "fori_loop", raise_index_error)
    assert control.fori_loop(0, 4, lambda i, x: x + i, 0) == 6


def test_while_loop_falls_back_to_python_when_jax_errors(monkeypatch) -> None:
    def raise_type_error(*args, **kwargs):
        del args, kwargs
        raise jax.errors.JAXTypeError("boom")

    monkeypatch.setattr(control.jax.lax, "while_loop", raise_type_error)
    assert control.while_loop(lambda x: x < 4, lambda x: x + 1, 1) == 4
