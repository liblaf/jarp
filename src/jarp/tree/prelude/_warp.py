import jax.tree_util as jtu
import warp as wp


def register_warp_array() -> None:
    jtu.register_static(wp.array)
