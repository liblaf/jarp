import logging

import jax
import jax.numpy as jnp
import pytest

from jarp.utils import suppress_jax_errors


def _trigger_tracer_integer_conversion() -> None:
    pred = jnp.asarray(1, dtype=jnp.bool_)

    def true_fun(i: int) -> int:
        return [10, 20][i]

    def false_fun(_i: int) -> int:
        return -1

    jax.lax.cond(
        pred,
        true_fun,
        false_fun,
        1,
    )


def test_suppress_jax_errors_logs_and_swallows_jax_errors(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("tests.suppress_jax_errors")
    with (
        caplog.at_level(logging.ERROR, logger=logger.name),
        suppress_jax_errors("suppressed", logger=logger),
    ):
        _trigger_tracer_integer_conversion()
    assert "suppressed" in caplog.text


def test_suppress_jax_errors_leaves_other_exceptions_alone() -> None:
    msg = "boom"
    with pytest.raises(RuntimeError, match="boom"), suppress_jax_errors():
        raise RuntimeError(msg)
