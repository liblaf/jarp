# PyTree Workflows

This guide covers the pieces of `jarp` that make mixed data-and-metadata trees
behave predictably under JAX.

## Choose Field Behavior Explicitly

```python
import jax.numpy as jnp
import jax.tree_util as jtu
import jarp


@jarp.define
class Example:
    data: object = jarp.array(default=0.0)
    label: str = jarp.static(default="")
    extra: object = jarp.auto(default="")


obj = Example()
leaves, _ = jtu.tree_flatten(obj)

obj.extra = jnp.zeros(())
leaves_with_extra, _ = jtu.tree_flatten(obj)
```

`data` always flattens as a JAX child. `label` always stays static. `extra`
follows the runtime value: a string stays static, while an array becomes a
dynamic child.

## Flatten A Tree Once And Reuse Its Structure

```python
import jax.numpy as jnp
import jarp


payload = {"a": jnp.zeros((3,)), "b": jnp.ones((4,)), "static": "foo"}
flat, structure = jarp.ravel(payload)

same_shape = {"a": jnp.ones((3,)), "b": jnp.zeros((4,)), "static": "foo"}
flat_again = structure.ravel(same_shape)
round_trip = structure.unravel(flat)
```

Use `ravel` when an optimizer, solver, or serialization step wants one vector
without losing the tree layout or static leaves. The returned
[`Structure`][jarp.tree.Structure] can flatten another compatible tree later or
rebuild the original layout from a flat vector.

## Wrap Foreign Objects As PyTrees

```python
import jax
import jax.numpy as jnp
import jarp


proxy = jarp.PyTreeProxy((jnp.zeros(()), "static"))
leaves, treedef = jax.tree.flatten(proxy)
restored = jax.tree.unflatten(treedef, leaves)
```

`PyTreeProxy` keeps the wrapper transparent while JAX traverses the wrapped
value. [`jarp.partial`][jarp.tree.partial] provides the same idea for partially
applied callables whose bound arguments should remain visible to tree
traversals.

`jarp.register_pytree_prelude()` performs the built-in one-time registrations
used by filtered JIT, including bound methods and `warp.array`. Most users only
need it when they want those registrations before using the higher-level
helpers that already call it.

## Register Non-`attrs` Classes

Use `jarp.tree.register_generic` when a class does not come from `attrs` or
when you want to spell out which fields are always data, always metadata, or
filtered at runtime.

See [the API reference for `jarp.tree`](../reference/jarp/tree/README.md),
[`jarp.tree.prelude`](../reference/jarp/tree/prelude/README.md), and
[`jarp.tree.codegen`](../reference/jarp/tree/codegen/README.md) for the exact
registration API and generated callback helpers.
