# Warp Interop

`jarp.warp` covers the boundary between JAX or NumPy arrays and NVIDIA Warp.
The simple case is array conversion. The more advanced case is exposing Warp
functions back to JAX through Warp's `jax_experimental` adapters.

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

## Expose Generic Warp Kernels To JAX

```python
from typing import Any

import jax.numpy as jnp
import jarp
import warp as wp


@jarp.warp.jax_kernel(
    arg_types_factory=lambda dtype: {"x": wp.array1d[dtype], "y": wp.array1d[dtype]}
)
@wp.kernel
def identity_kernel(x: wp.array1d[Any], y: wp.array1d[Any]) -> None:
    tid = wp.tid()
    y[tid] = x[tid]


(y,) = identity_kernel(jnp.ones((7,), jnp.float32))
```

This pattern chooses a Warp overload from the runtime JAX dtypes. The generic
`jax_callable` wrapper follows the same idea, but builds a callable factory
instead of an overload signature.

The repository's `jax_callable` and `jax_kernel` tests are skipped when CUDA is
unavailable, so these adapters should be treated as Warp-runtime features whose
availability depends on the local Warp setup.

## Precision-Aware Warp Types

`jarp.warp.types.floating`, `vecN`, and `matMN` follow JAX's active
`jax_enable_x64` setting. Use them when Warp dtypes should match the precision
mode already chosen by the surrounding JAX program.

See [`jarp.warp`](../reference/jarp/warp/README.md) and
[`jarp.warp.types`](../reference/jarp/warp/types.md) for the full API surface.
