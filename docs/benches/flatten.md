# Custom PyTree Nodes

This benchmark measures flatten and unflatten performance for custom PyTree
nodes with seven data fields and three static fields.

[Source Code](https://github.com/liblaf/jarp/blob/main/benches/flatten.py)

|          Method          | Converter | Flatten | Unflatten |    Total |
| :----------------------: | :-------: | ------: | --------: | -------: |
|      `jarp.define`       |    no     | 0.48 µs |   0.30 µs |  0.78 µs |
|      `jarp.define`       |    yes    | 0.49 µs |   0.65 µs |  1.14 µs |
| `jtu.register_dataclass` |    no     | 0.42 µs |   0.77 µs |  1.19 µs |
| `jtu.register_dataclass` |    yes    | 0.41 µs |  25.23 µs | 25.65 µs |
|     `equinox.Module`     |    no     | 0.89 µs |   0.76 µs |  1.65 µs |
|     `equinox.Module`     |    yes    | 0.90 µs |   0.79 µs |  1.69 µs |

## `jarp.define`

`jarp` gets its unflatten speed from generated Python callbacks for each
registered class.

- **Bypassing `__init__`**: The generated unflatten function creates a new
  instance with `object.__new__` and populates fields directly. Converters and
  validators do not rerun during unflattening.
- **Assignment strategy**: For simple classes, generated code can assign fields
  directly. When attribute setters may interfere, it can use
  `object.__setattr__` instead.

## `jax.tree_util.register_dataclass`

When converters are present, `jtu` reconstructs objects by calling `__init__`,
which reruns converters such as `jnp.asarray` while restoring a tree node.

## `equinox.Module`

Equinox is consistent regardless of converters, with a higher baseline than
the generated `jarp` callbacks in this benchmark.

## Test Environment

```text
python==3.14.3
jax==0.10.0
liblaf-jarp==0.1.10.dev9+g99e249b88
equinox==0.13.8
```
