import dataclasses
import timeit
from typing import Any

import attrs
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import Array

import jarp


@jtu.register_dataclass
@dataclasses.dataclass
class TreeJaxWithoutConverter:
    a: Array = dataclasses.field(default_factory=lambda: jnp.zeros(()))
    b: Array = dataclasses.field(default_factory=lambda: jnp.zeros(()))
    c: Array = dataclasses.field(default_factory=lambda: jnp.zeros(()))
    d: Array = dataclasses.field(default_factory=lambda: jnp.zeros(()))
    e: Array = dataclasses.field(default_factory=lambda: jnp.zeros(()))
    f: Array = dataclasses.field(default_factory=lambda: jnp.zeros(()))
    g: Array = dataclasses.field(default_factory=lambda: jnp.zeros(()))
    h: str = dataclasses.field(default="", metadata={"static": True})
    i: str = dataclasses.field(default="", metadata={"static": True})
    j: str = dataclasses.field(default="", metadata={"static": True})


@attrs.define
class TreeJaxWithConverter:
    a: Array = attrs.field(converter=jnp.asarray, factory=lambda: jnp.zeros(()))
    b: Array = attrs.field(converter=jnp.asarray, factory=lambda: jnp.zeros(()))
    c: Array = attrs.field(converter=jnp.asarray, factory=lambda: jnp.zeros(()))
    d: Array = attrs.field(converter=jnp.asarray, factory=lambda: jnp.zeros(()))
    e: Array = attrs.field(converter=jnp.asarray, factory=lambda: jnp.zeros(()))
    f: Array = attrs.field(converter=jnp.asarray, factory=lambda: jnp.zeros(()))
    g: Array = attrs.field(converter=jnp.asarray, factory=lambda: jnp.zeros(()))
    h: str = attrs.field(default="", metadata={"static": True})
    i: str = attrs.field(default="", metadata={"static": True})
    j: str = attrs.field(default="", metadata={"static": True})


jtu.register_dataclass(
    TreeJaxWithConverter,
    data_fields=("a", "b", "c", "d", "e", "f", "g"),
    meta_fields=("h", "i", "j"),
)


@jarp.define
class TreeJarpWithoutConverter:
    a: Array = jarp.field(factory=lambda: jnp.zeros(()))
    b: Array = jarp.field(factory=lambda: jnp.zeros(()))
    c: Array = jarp.field(factory=lambda: jnp.zeros(()))
    d: Array = jarp.field(factory=lambda: jnp.zeros(()))
    e: Array = jarp.field(factory=lambda: jnp.zeros(()))
    f: Array = jarp.field(factory=lambda: jnp.zeros(()))
    g: Array = jarp.field(factory=lambda: jnp.zeros(()))
    h: str = jarp.static(default="")
    i: str = jarp.static(default="")
    j: str = jarp.static(default="")


@jarp.define
class TreeJarpWithConverter:
    a: Array = jarp.field(converter=jnp.asarray, factory=lambda: jnp.zeros(()))
    b: Array = jarp.field(converter=jnp.asarray, factory=lambda: jnp.zeros(()))
    c: Array = jarp.field(converter=jnp.asarray, factory=lambda: jnp.zeros(()))
    d: Array = jarp.field(converter=jnp.asarray, factory=lambda: jnp.zeros(()))
    e: Array = jarp.field(converter=jnp.asarray, factory=lambda: jnp.zeros(()))
    f: Array = jarp.field(converter=jnp.asarray, factory=lambda: jnp.zeros(()))
    g: Array = jarp.field(converter=jnp.asarray, factory=lambda: jnp.zeros(()))
    h: str = jarp.static(default="")
    i: str = jarp.static(default="")
    j: str = jarp.static(default="")


class TreeEquinoxWithoutConverter(eqx.Module):
    a: Array = eqx.field(default_factory=lambda: jnp.zeros(()))
    b: Array = eqx.field(default_factory=lambda: jnp.zeros(()))
    c: Array = eqx.field(default_factory=lambda: jnp.zeros(()))
    d: Array = eqx.field(default_factory=lambda: jnp.zeros(()))
    e: Array = eqx.field(default_factory=lambda: jnp.zeros(()))
    f: Array = eqx.field(default_factory=lambda: jnp.zeros(()))
    g: Array = eqx.field(default_factory=lambda: jnp.zeros(()))
    h: str = eqx.field(default="", static=True)
    i: str = eqx.field(default="", static=True)
    j: str = eqx.field(default="", static=True)


class TreeEquinoxWithConverter(eqx.Module):
    a: Array = eqx.field(default_factory=lambda: jnp.zeros(()), converter=jnp.asarray)
    b: Array = eqx.field(default_factory=lambda: jnp.zeros(()), converter=jnp.asarray)
    c: Array = eqx.field(default_factory=lambda: jnp.zeros(()), converter=jnp.asarray)
    d: Array = eqx.field(default_factory=lambda: jnp.zeros(()), converter=jnp.asarray)
    e: Array = eqx.field(default_factory=lambda: jnp.zeros(()), converter=jnp.asarray)
    f: Array = eqx.field(default_factory=lambda: jnp.zeros(()), converter=jnp.asarray)
    g: Array = eqx.field(default_factory=lambda: jnp.zeros(()), converter=jnp.asarray)
    h: str = eqx.field(default="", static=True)
    i: str = eqx.field(default="", static=True)
    j: str = eqx.field(default="", static=True)


def bench_flatten(obj: Any) -> float:
    timer = timeit.Timer(lambda: jax.tree.flatten(obj))
    number: int
    time_taken: float
    number, time_taken = timer.autorange()
    return time_taken / number


def bench_unflatten(obj: Any) -> float:
    leaves: list[Any]
    treedef: Any
    leaves, treedef = jax.tree.flatten(obj)
    timer = timeit.Timer(lambda: jax.tree.unflatten(treedef, leaves))
    number: int
    time_taken: float
    number, time_taken = timer.autorange()
    return time_taken / number


def print_elapsed(name: str, flatten: float, unflatten: float) -> None:
    print(f"{name} flatten: {flatten * 1e6:.2f} µs")
    print(f"{name} unflatten: {unflatten * 1e6:.2f} µs")
    print(f"{name} total: {(flatten + unflatten) * 1e6:.2f} µs")


def main() -> None:
    obj_jax_no_converter = TreeJaxWithoutConverter()
    flatten_elapsed: float = bench_flatten(obj_jax_no_converter)
    unflatten_elapsed: float = bench_unflatten(obj_jax_no_converter)
    print_elapsed("JAX dataclass w/o converter", flatten_elapsed, unflatten_elapsed)

    obj_jax_converter = TreeJaxWithConverter()
    flatten_elapsed = bench_flatten(obj_jax_converter)
    unflatten_elapsed = bench_unflatten(obj_jax_converter)
    print_elapsed("JAX dataclass w/ converter", flatten_elapsed, unflatten_elapsed)

    obj_jarp_no_converter = TreeJarpWithoutConverter()
    flatten_elapsed: float = bench_flatten(obj_jarp_no_converter)
    unflatten_elapsed: float = bench_unflatten(obj_jarp_no_converter)
    print_elapsed("Jarp w/o converter", flatten_elapsed, unflatten_elapsed)

    obj_jarp_converter = TreeJarpWithConverter()
    flatten_elapsed: float = bench_flatten(obj_jarp_converter)
    unflatten_elapsed: float = bench_unflatten(obj_jarp_converter)
    print_elapsed("Jarp w/ converter", flatten_elapsed, unflatten_elapsed)

    obj_eqx_no_converter = TreeEquinoxWithoutConverter()
    flatten_elapsed: float = bench_flatten(obj_eqx_no_converter)
    unflatten_elapsed: float = bench_unflatten(obj_eqx_no_converter)
    print_elapsed("Equinox w/o converter", flatten_elapsed, unflatten_elapsed)

    obj_eqx_converter = TreeEquinoxWithConverter()
    flatten_elapsed: float = bench_flatten(obj_eqx_converter)
    unflatten_elapsed: float = bench_unflatten(obj_eqx_converter)
    print_elapsed("Equinox w/ converter", flatten_elapsed, unflatten_elapsed)


if __name__ == "__main__":
    main()
