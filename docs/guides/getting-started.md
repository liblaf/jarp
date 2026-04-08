# Getting Started

`jarp` is for PyTrees that contain both traceable arrays and ordinary Python
metadata. The common pattern is to describe that split once with field
specifiers, then reuse it everywhere else.

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


@jarp.filter_jit
def normalize(batch: Batch) -> Batch:
    centered = batch.values - jnp.mean(batch.values)
    return Batch(values=centered, label=batch.label)


batch = Batch(values=jnp.array([1.0, 2.0, 3.0]), label="train")
result = normalize(batch)
```

`array()` marks values that should stay on the dynamic side of the partition.
`static()` marks metadata that should stay out of the dynamic leaves.
`auto()` is the middle ground: it decides at flatten time whether the current
value behaves like data or metadata.

[`filter_jit`][jarp.filter_jit] uses the same split for ordinary call
arguments, so a function can accept strings, callables, or other metadata
inside the same tree as JAX arrays without manual tree surgery.

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

If you already have a compatible tree, [`Structure.ravel`][jarp.tree.Structure.ravel]
can flatten it again and [`Structure.unravel`][jarp.tree.Structure.unravel]
will accept an already-matching tree unchanged.

## Retry Selected Control-Flow Errors Eagerly

[`jarp.lax`][jarp.lax] tries `jax.lax` first and reruns the same callbacks in
plain Python when JAX raises the tracing or indexing errors covered by
[`suppress_jax_errors`][jarp.utils.suppress_jax_errors].

```python
import jarp


value = jarp.lax.while_loop(
    lambda state: state[0] < 3,
    lambda state: (state[0] + 1, state[1] + [10, 20, 30][state[0]]),
    (0, 0),
)
```

For the control-flow helpers and the cached Python fallback in
[`fallback_jit`][jarp.fallback_jit], continue with
[Call wrappers](call-wrappers.md).

## Next Steps

- Read [Call wrappers](call-wrappers.md) for `filter_jit`,
  `fallback_jit`, and `jarp.lax`.
- Read [PyTree workflows](pytree-workflows.md) for `auto`, `PyTreeProxy`, and
  custom registration helpers.
- Read [Warp interop](warp.md) for `to_warp`, `jax_callable`, and
  `jax_kernel`.
- Use the [API reference](../reference/README.md) when you need exact
  signatures.
