import timeit
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

import jarp

# jax.config.update("jax_platforms", "cpu")


def fun(inputs: Any = None) -> Any:
    return inputs


def bench(fun: Callable[..., Any]) -> float:
    for _ in range(2000):  # warmup
        fun()
    timer = timeit.Timer(fun)
    number: int = 2000
    time_taken: float = timer.timeit(number)
    return time_taken / number


def print_elapsed(name: str, time_taken: float) -> None:
    print(f"{name}: {time_taken * 1e6:.2f} µs")


def main() -> None:
    fun_jax: Callable[..., Any] = jax.jit(fun)
    fun_jarp: Callable[..., Any] = jarp.jit(fun, filter=True)
    fun_eqx: Callable[..., Any] = eqx.filter_jit(fun)
    print_elapsed("JAX JIT", bench(lambda: jax.block_until_ready(fun_jax())))
    print_elapsed("JARP JIT", bench(lambda: jax.block_until_ready(fun_jarp())))
    print_elapsed("Equinox JIT", bench(lambda: jax.block_until_ready(fun_eqx())))

    inputs: list = [[jnp.zeros(())] * 10] * 10
    print_elapsed(
        "JAX JIT (with inputs)", bench(lambda: jax.block_until_ready(fun_jax(inputs)))
    )
    print_elapsed(
        "JARP JIT (with inputs)",
        bench(lambda: jax.block_until_ready(fun_jarp(inputs))),
    )
    print_elapsed(
        "Equinox JIT (with inputs)",
        bench(lambda: jax.block_until_ready(fun_eqx(inputs))),
    )


if __name__ == "__main__":
    main()
