from jax._src.tree_util import _registry


def in_registry(cls: type) -> bool:
    return cls in _registry
