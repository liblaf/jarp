import logging

import jax
import pytest

from jarp.utils import suppress_jax_errors


def test_suppress_jax_errors_logs_and_continues(caplog) -> None:
    logger = logging.getLogger("tests.suppress")

    with caplog.at_level(logging.ERROR, logger=logger.name):
        with suppress_jax_errors("suppressed", logger=logger):
            raise jax.errors.JAXTypeError("boom")

    assert "suppressed" in caplog.text
    assert "boom" in caplog.text


def test_suppress_jax_errors_does_not_hide_other_errors() -> None:
    with pytest.raises(RuntimeError, match="boom"):
        with suppress_jax_errors():
            raise RuntimeError("boom")
