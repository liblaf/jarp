# Warp Interop

`jarp.warp` covers the boundary between JAX or NumPy arrays and NVIDIA Warp.
The simple case is array conversion. The more advanced case is rebuilding Warp
callables and kernel overloads from the runtime JAX dtypes.

## Convert Arrays To Warp

```python
from typing import Any

import jax.numpy as jnp
import jarp


scalar = jarp.to_warp(jnp.zeros((7,), jnp.float32))
vector = jarp.to_warp(jnp.zeros((5, 3), jnp.float32), (-1, Any))
matrix = jarp.to_warp(jnp.zeros((2, 3, 3), jnp.float32), (-1, -1, Any))
```

Passing `(-1, Any)` asks `jarp` to infer the vector length from the trailing
dimension. Passing `(-1, -1, Any)` does the same for matrix row and column
counts. The scalar dtype defaults to the array dtype when the tuple ends in
`Any` or `None`.

For JAX arrays, `requires_grad=True` is applied after `warp.from_jax(...)` so
the resulting `warp.array` can opt into Warp gradients when needed.

## Expose Generic Warp Adapters To JAX

[`jax_callable`][jarp.warp.jax_callable] can treat its input function as a
factory keyed by the runtime Warp scalar dtypes inferred from the JAX
arguments. The wrapper caches each factory result by dtype signature.

[`jax_kernel`][jarp.warp.jax_kernel] performs the related overload-selection
step for Warp kernels when you provide `arg_types_factory`.

The repository proves the adapter wiring and dtype dispatch in unit tests, but
running real Warp kernels still depends on the local Warp runtime and hardware
setup.

## Precision-Aware Warp Types

[`jarp.warp.types.floating`][jarp.warp.types.floating], `vecN`, and `matMN`
follow JAX's active `jax_enable_x64` setting. Use them when Warp dtypes should
match the precision mode already chosen by the surrounding JAX program.

See [`jarp.warp`](../reference/jarp/warp/README.md) and
[`jarp.warp.types`](../reference/jarp/warp/types.md) for the full API surface.
