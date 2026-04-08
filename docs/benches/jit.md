# Filtered Call Wrappers

This benchmark compares the steady-state invocation overhead around a no-op
function after warmup. `jax.jit` and `equinox.filter_jit` measure compiled call
overhead. `jarp.filter_jit` measures the cost of partitioning mixed inputs and
recombining outputs on the same callable shape.

[Source Code](https://github.com/liblaf/jarp/blob/main/benches/jit.py)

|        Method        | No PyTree | Complex PyTree |
| :------------------: | --------: | -------------: |
|      `jax.jit`       |   7.36 µs |      768.07 µs |
|  `jarp.filter_jit`   |  11.09 µs |      933.26 µs |
| `equinox.filter_jit` | 292.72 µs |     1149.43 µs |

## JAX

`jax.jit` provides the compiled-call baseline. It also imposes the strictest
input requirements: leaves must be JAX-friendly values or JAX will raise while
tracing.

## JARP

`jarp.filter_jit` introduces a lightweight filtering mechanism. It partitions
the call into:

- **Dynamic leaves**: JAX arrays and `None` placeholders.
- **Static leaves**: Other values, which are stored as metadata and stitched
  back into the original call shape.

That lets users pass mixed PyTrees through one callable boundary without manual
partitioning. The overhead for this convenience is small in the benchmark.

## Equinox

`equinox.filter_jit` is the closest comparison point for mixed-tree call
wrappers. In this microbenchmark, its invocation overhead is significantly
higher.

## Test Environment

```text
python==3.14.2
jax==0.9.0
jarp==0.1.0
equinox==0.13.2
```
