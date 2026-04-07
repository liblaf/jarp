# Getting Started

`jarp` is for PyTrees that contain both traceable arrays and ordinary Python
metadata. The common pattern is to describe that split once with field
specifiers, then let `jarp` carry it through JAX transforms.

## Install

```bash
uv add jarp
```

Optional extras install CUDA-enabled JAX wheels that match the environment:

```bash
uv add 'jarp[cuda12]'
uv add 'jarp[cuda13]'
```

## Define A PyTree-Friendly Class

```python
import jax.numpy as jnp
import jarp


@jarp.define
class Batch:
    values: object = jarp.array()
    label: str = jarp.static()


@jarp.jit(filter=True)
def normalize(batch: Batch) -> Batch:
    centered = batch.values - jnp.mean(batch.values)
    return Batch(values=centered, label=batch.label)


batch = Batch(values=jnp.array([1.0, 2.0, 3.0]), label="train")
result = normalize(batch)
```

`array()` marks values that should stay dynamic under JAX transforms.
`static()` marks metadata that should stay out of tracing. `auto()` is the
middle ground: it decides at flatten time whether the current value behaves
like data or metadata.

`jarp.jit(filter=True)` uses the same split for ordinary call arguments, so a
function can accept strings, callables, or other metadata inside the same tree
as JAX arrays without manually wiring `static_argnums`.

## Flatten Mixed Trees Into One Vector

```python
import jax.numpy as jnp
import jarp


payload = {"a": jnp.zeros((3,)), "b": jnp.ones((4,)), "static": "foo"}
flat, structure = jarp.ravel(payload)
round_trip = structure.unravel(flat)
```

`flat` contains only the dynamic leaves. `structure` keeps the tree definition,
static leaves, and reshape offsets needed to rebuild compatible values later.

If you already have a compatible tree, `structure.ravel(tree)` and
`structure.unravel(tree)` treat it as an identity round trip after validating
the recorded structure.

## Optional Eager Control Flow

`jarp.lax` mirrors a small slice of `jax.lax` and adds a `jit=` switch for the
branching and loop wrappers. That lets the same callback structure run eagerly
in Python during debugging:

```python
import jarp


value = jarp.lax.while_loop(
    lambda x: x < 3,
    lambda x: x + 1,
    0,
    jit=False,
)
```

## Next Steps

- Read [PyTree workflows](pytree-workflows.md) for `auto`, `PyTreeProxy`, and
  custom registration helpers.
- Read [Warp interop](warp.md) for `to_warp`, `jax_callable`, and
  `jax_kernel`.
- Use the [API reference](../reference/README.md) when you need exact
  signatures.
