from typing import Any

import jax.numpy as jnp
from jax import Array

from jarp import filter_jit


def test_filter_jit_preserves_static_metadata_and_wrapped_attrs() -> None:
    @filter_jit
    def pack(x: Array, label: str = "tag") -> dict[str, Any]:
        return {"x": x + 1, "label": label}

    result = pack(jnp.array([1, 2]), label="tag")
    assert result["x"].tolist() == [2, 3]
    assert result["label"] == "tag"
    assert pack.__name__ == "pack"


def test_filter_jit_supports_deferred_decoration_on_methods() -> None:
    class Scale:
        def __init__(self, factor: int) -> None:
            self.factor = factor

        @filter_jit()
        def apply(self, x: Array, bias: int = 0) -> Array:
            return x * self.factor + bias

    result = Scale(3).apply(jnp.array([1, 2]), bias=1)
    assert result.tolist() == [4, 7]
