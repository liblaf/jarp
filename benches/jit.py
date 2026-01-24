import timeit
from collections.abc import Callable

import equinox as eqx
import jax

import jarp

type NOP = Callable[[], None]


def fun() -> None: ...


def bench(fun: NOP) -> float:
    fun()  # warmup
    timer = timeit.Timer(fun)
    number: int
    time_taken: float
    number, time_taken = timer.autorange()
    return time_taken / number


def main() -> None:
    fun_jax: NOP = jax.jit(fun)
    fun_jarp: NOP = jarp.jit(fun, filter=True)
    fun_eqx: NOP = eqx.filter_jit(fun)
    print("JAX JIT:", bench(fun_jax) * 1e6, "µs")
    print("JARP JIT:", bench(fun_jarp) * 1e6, "µs")
    print("Equinox JIT:", bench(fun_eqx) * 1e6, "µs")


if __name__ == "__main__":
    main()
