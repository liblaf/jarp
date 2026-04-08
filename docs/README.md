# jarp

`jarp` keeps mixed JAX PyTrees usable across function boundaries, `attrs`, and
NVIDIA Warp. The package is intentionally small: it focuses on filtered call
wrappers, PyTree-friendly class definitions, round-trippable flattening, a
handful of control-flow wrappers, and Warp adapters that fit JAX-first
workflows.

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
```

## Read By Workflow

- [Getting started](guides/getting-started.md) for installation, the core field
  specifiers, and the first filtered call wrapper.
- [Call wrappers](guides/call-wrappers.md) for `filter_jit`,
  `fallback_jit`, and the `jarp.lax` helpers.
- [PyTree workflows](guides/pytree-workflows.md) for `define`, `auto`,
  `ravel`, `PyTreeProxy`, and custom registration helpers.
- [Warp interop](guides/warp.md) for `to_warp`, generic Warp adapters, and
  dtype helpers.
- [API reference](reference/README.md) for exact signatures and module-level
  details.
- [Benchmarks](benches/jit.md) for the current wrapper-overhead and PyTree
  registration measurements.
