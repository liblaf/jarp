# Call Wrappers

`liblaf.jarp` exposes two callable wrappers and a small `lax` compatibility
layer for mixed JAX-and-Python code.

## Partition Mixed Call Arguments

`filter_jit` splits each call into dynamic array leaves and
static metadata, rebuilds the original call shape, and partitions the return
value again on the way out.

```python
from typing import Any

import jax.numpy as jnp
from jax import Array
from liblaf import jarp


@jarp.filter_jit
def pack(x: Array, label: str = "tag") -> dict[str, Any]:
    return {"x": x + 1, "label": label}


result = pack(jnp.array([1, 2]), label="train")
```

The wrapper also preserves method binding, so `@filter_jit()` works on instance
methods as well as free functions.

## Cache Python Fallbacks By Metadata Shape

`fallback_jit` starts with the same partitioned call path as `filter_jit`. If
that path raises `jax.errors.JAXTypeError` or `jax.errors.JAXIndexError`,
`jarp` logs the exception, marks the current static-metadata signature as
unsupported, and reuses the direct Python call path for later calls with the
same metadata.

Use it when the same callable sometimes works cleanly with JAX-style inputs but
needs a stable eager fallback for particular metadata layouts.

## Retry `jax.lax` Helpers Eagerly

`jarp.lax` wraps `jax.lax.cond`, `jax.lax.switch`, `jax.lax.fori_loop`, and
`jax.lax.while_loop`. Each wrapper tries the JAX primitive first and reruns
eagerly if JAX raises one of the errors handled by `LaxWrapper`.

```python
from liblaf import jarp


state = jarp.lax.while_loop(
    lambda value: value[0] < 3,
    lambda value: (value[0] + 1, value[1] + [10, 20, 30][value[0]]),
    (0, 0),
)
```

On the eager fallback path, `jarp.lax.switch` clamps the branch index into
range before dispatch.

## Collapse Ordered Conditions To An Index

`first_true_index` turns an ordered list of scalar or array conditions into a
JAX integer array. It returns the first matching condition index at each
position, and uses `len(condlist)` where no condition matches.

```python
import jax.numpy as jnp
from liblaf import jarp


labels = jarp.first_true_index(
    [
        jnp.array([False, True, False, False]),
        jnp.array([True, True, False, False]),
        jnp.array([True, False, True, False]),
    ]
)
```

## Preserve Primitive Metadata

The public `jarp.lax` helpers are `LaxWrapper` instances. They keep the wrapped
`jax.lax` primitive available through `__wrapped__`, preserve the primitive
signature for inspection, and cache metadata signatures that should skip
directly to the Python fallback after a supported failure.

`LaxWrapper` copies ordinary function metadata when it is available, but it
does not require it. Callable objects with only `__call__` still work as the
wrapped JAX-side callable.
