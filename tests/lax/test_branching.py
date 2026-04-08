import jax

import jarp.lax._control as control


def test_cond_falls_back_to_python_when_jax_errors(monkeypatch) -> None:
    def raise_type_error(*args, **kwargs):
        del args, kwargs
        raise jax.errors.JAXTypeError("boom")

    monkeypatch.setattr(control.jax.lax, "cond", raise_type_error)
    assert control.cond(True, lambda x: x + 1, lambda x: x - 1, 2) == 3
    assert control.cond(False, lambda x: x + 1, lambda x: x - 1, 2) == 1


def test_switch_clamps_index_in_python_fallback(monkeypatch) -> None:
    def raise_index_error(*args, **kwargs):
        del args, kwargs
        raise jax.errors.JAXIndexError("boom")

    monkeypatch.setattr(control.jax.lax, "switch", raise_index_error)
    branches = (lambda x: x - 1, lambda x: x + 1)

    assert control.switch(-5, branches, 3) == 2
    assert control.switch(99, branches, 3) == 4
