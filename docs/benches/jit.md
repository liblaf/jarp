# Just-in-time compilation

This benchmark measures the invocation overhead of a "no-op" JIT-compiled function. Trace overhead is excluded by warming up the JIT cache.

|         Method          | Overhead |
| :---------------------: | -------: |
|        `jax.jit`        |   6.6 µs |
| `jarp.jit(filter=True)` |  10.3 µs |
|  `equinox.filter_jit`   | 284.0 µs |

## JAX

`jax.jit` provides the baseline performance. However, it enforces strict requirements on arguments: all leaves in the PyTree must be JAX arrays (or compatible types), otherwise JAX will try to trace them and raise an error.

## JARP

`jarp.jit(filter=True)` introduces a lightweight filtering mechanism. It automatically partitions the arguments into:

- **Dynamic leaves**: JAX arrays, which are passed to the JIT-compiled function.
- **Static leaves**: Non-array values, which are treated as static data (part of the compiled function's constant context).

This allows users to pass arbitrary PyTrees containing mixed data types to JIT-compiled functions without manual intervention. The overhead for this convenience is small (4 µs).

## Equinox

`equinox.filter_jit` offers similar functionality to `jarp.jit`, separating dynamic and static parts of a PyTree. However, in this microbenchmark, its invocation overhead is significantly higher.

## Test Environment

```
python==3.14.2
jax==0.9.0
jarp==0.1.0
equinox==0.13.2
```
