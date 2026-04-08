import jax.numpy as jnp

from jarp import filter_jit


def test_filter_jit_preserves_static_metadata_and_wrapped_attrs() -> None:
    @filter_jit
    def pack(x, label="tag"):
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
        def apply(self, x, bias=0):
            return x * self.factor + bias

    result = Scale(3).apply(jnp.array([1, 2]), bias=1)
    assert result.tolist() == [4, 7]
