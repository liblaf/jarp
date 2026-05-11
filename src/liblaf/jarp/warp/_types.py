from collections.abc import Sequence
from typing import Any, Literal

type ArgTypes = dict[str, WarpDType] | list[WarpDType] | None
type ShapeLike = int | Sequence[int]
type VmapMethod = Literal["broadcast_all", "expand_dims", "sequential"]
type WarpDType = Any
type WarpScalarDType = Any
