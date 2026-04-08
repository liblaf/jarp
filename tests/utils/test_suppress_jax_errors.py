import logging

import jax
import jax.numpy as jnp
import pytest

from jarp.utils import suppress_jax_errors


def _trigger_tracer_integer_conversion() -> None:
    jax.lax.cond(
        jnp.array(True),
        lambda i: [10, 20][i],
        lambda i: -1,
        1,
    )


def test_suppress_jax_errors_logs_and_swallows_jax_errors(caplog) -> None:
    logger = logging.getLogger("tests.suppress_jax_errors")
    with caplog.at_level(logging.ERROR, logger=logger.name):
        with suppress_jax_errors("suppressed", logger=logger):
            _trigger_tracer_integer_conversion()
    assert "suppressed" in caplog.text


def test_suppress_jax_errors_leaves_other_exceptions_alone() -> None:
    with pytest.raises(RuntimeError, match="boom"):
        with suppress_jax_errors():
            raise RuntimeError("boom")
