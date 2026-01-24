# Custom PyTree Nodes

This benchmark measures the flatten and unflatten performance of custom PyTree nodes defined with 7 data fields and 3 static fields.

|          Method          | Converter | Flatten | Unflatten |    Total |
| :----------------------: | :-------: | ------: | --------: | -------: |
|      `jarp.define`       |    w/o    | 0.56 µs |   0.34 µs |  0.90 µs |
|      `jarp.define`       |    w/     | 0.57 µs |   0.85 µs |  1.43 µs |
| `jtu.register_dataclass` |    w/o    | 0.51 µs |   0.86 µs |  1.37 µs |
| `jtu.register_dataclass` |    w/     | 0.52 µs |  30.10 µs | 30.62 µs |
|     `equinox.Module`     |    w/o    | 1.66 µs |   1.32 µs |  2.98 µs |
|     `equinox.Module`     |    w/     | 1.64 µs |   1.32 µs |  2.97 µs |

### `jarp.define`

`jarp` achieves the highest performance by generating specialized Python code for each class.

- **Bypassing `__init__`**: The generated unflatten function creates a new instance using `object.__new__` and populates fields directly. This completely bypasses `__init__`, preventing converters and validators from running redundantly during unflattening. This is why `jarp` avoids the massive 30 µs penalty seen in `jtu.register_dataclass` when converters are used.
- **Micro-optimizations**:
  - In the "w/o" case, `jarp` detects that it can safe use direct assignment (`obj.x = val`), resulting in the fastest unflatten time (0.34 µs).
  - In the "w/" case, `jarp` may fall back to `object.__setattr__` to bypass potentially overridden attribute setters. This adds a tiny overhead (+0.5 µs), but it is still **~35x faster** than JAX's approach.

### `jax.tree_util.register_dataclass`

- **Impact of Converters**: When converters are present, `jtu`'s unflattening time explodes to 30.10 µs. This is because `jax` reconstructs objects by calling `__init__`, forcibly re-executing all converters (like `jnp.asarray`) even when restoring an internal tree node.

### `equinox.Module`

- **Overhead**: Equinox shows consistent performance regardless of converters, but has a higher baseline overhead (3 µs total) compared to `jarp` (1 µs).
