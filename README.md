<div align="center" markdown>
<a name="readme-top"></a>

![jarp](https://socialify.git.ci/liblaf/jarp/image?description=1&forks=1&issues=1&language=1&name=1&owner=1&pattern=Transparent&pulls=1&stargazers=1&theme=Auto)

**[Explore the docs »](https://liblaf.github.io/jarp/)**

<!-- tangerine-start: badges/python.md -->

[![Codecov](https://codecov.io/gh/liblaf/jarp/graph/badge.svg)](https://codecov.io/gh/liblaf/jarp)
[![MegaLinter](https://github.com/liblaf/jarp/actions/workflows/shared-mega-linter.yaml/badge.svg)](https://github.com/liblaf/jarp/actions/workflows/shared-mega-linter.yaml)
[![Python / Docs](https://github.com/liblaf/jarp/actions/workflows/python-docs.yaml/badge.svg)](https://github.com/liblaf/jarp/actions/workflows/python-docs.yaml)
[![Python / Test](https://github.com/liblaf/jarp/actions/workflows/python-test.yaml/badge.svg)](https://github.com/liblaf/jarp/actions/workflows/python-test.yaml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/liblaf/jarp/main.svg)](https://results.pre-commit.ci/latest/github/liblaf/jarp/main)
[![CodSpeed Badge](https://img.shields.io/endpoint?url=https://codspeed.io/badge.json)](https://codspeed.io/liblaf/jarp)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/jarp?logo=PyPI&label=Downloads)](https://pypi.org/project/jarp)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/jarp?logo=Python&label=Python)](https://pypi.org/project/jarp)
[![PyPI - Version](https://img.shields.io/pypi/v/jarp?logo=PyPI&label=PyPI)](https://pypi.org/project/jarp)

<!-- tangerine-end -->

[Changelog](https://github.com/liblaf/jarp/blob/main/CHANGELOG.md) · [Report Bug](https://github.com/liblaf/jarp/issues) · [Request Feature](https://github.com/liblaf/jarp/issues)

![Rule](https://cdn.jsdelivr.net/gh/andreasbm/readme/assets/lines/rainbow.png)

</div>

`jarp` is a small utility library for JAX code that mixes traceable arrays with
ordinary Python metadata, plus a few helpers for crossing into NVIDIA Warp.
It focuses on the points where a plain `jax.jit` or basic PyTree registration
starts to feel too rigid:

- `jarp.jit(filter=True)` partitions mixed PyTrees into dynamic array leaves and
  static metadata automatically.
- `jarp.define`, `jarp.frozen`, `jarp.array()`, `jarp.static()`, and
  `jarp.auto()` make `attrs` classes flatten the way JAX code expects.
- `jarp.ravel` turns the dynamic leaves of a tree into one flat vector and
  returns a reusable `Structure` for round trips.
- `jarp.to_warp`, `jarp.warp.jax_callable`, and `jarp.warp.jax_kernel` cover
  the common JAX-to-Warp interop paths.
- `jarp.lax` mirrors a small slice of `jax.lax` with optional eager fallbacks
  for debugging or non-jitted execution.

## 📦 Installation

> [!NOTE]
> `jarp` requires Python 3.12 or newer.

Install the published package with `uv`:

```bash
uv add jarp
```

If you want a CUDA-enabled JAX extra, pick the matching wheel set:

```bash
uv add 'jarp[cuda12]'
uv add 'jarp[cuda13]'
```

## 🚀 Quick Start

This example shows the core workflow: define a mixed data-and-metadata PyTree
once, then compile functions against it without manual `static_argnums`
plumbing.

```python
import jax.numpy as jnp
import jarp


@jarp.define
class Batch:
    values: object = jarp.array()
    label: str = jarp.static()


@jarp.jit(filter=True)
def normalize(batch: Batch) -> Batch:
    centered = batch.values - jnp.mean(batch.values)
    return Batch(values=centered, label=batch.label)


batch = Batch(values=jnp.array([1.0, 2.0, 3.0]), label="train")
result = normalize(batch)
```

The array payload stays traceable, while the string label remains static
metadata.

`jarp.ravel` handles the other common workflow: flatten only the dynamic leaves
into one vector and keep enough structure around to rebuild the tree later.

```python
import jax.numpy as jnp
import jarp


payload = {"a": jnp.zeros((3,)), "b": jnp.ones((4,)), "static": "foo"}
flat, structure = jarp.ravel(payload)
round_trip = structure.unravel(flat)
```

When a JAX or NumPy pipeline needs to cross into Warp, `jarp.to_warp` can infer
vector and matrix dtypes from trailing dimensions:

```python
from typing import Any

import jax.numpy as jnp
import jarp


arr_wp = jarp.to_warp(jnp.zeros((5, 3), jnp.float32), (-1, Any))
```

For broader PyTree traversal helpers, see `jarp.PyTreeProxy`,
`jarp.partial`, `jarp.tree.register_generic`, and the lower-level
`jarp.tree.codegen` module.

## 🛠️ Local Development

Clone the repository, sync the workspace, and use `nox` for the maintained
automation surface:

```bash
git clone https://github.com/liblaf/jarp.git
cd jarp
uv sync --all-groups
nox --list-sessions
nox --tags test
```

To build the documentation site locally:

```bash
uv run zensical build
```

## 📚 Documentation

- [Documentation site](https://liblaf.github.io/jarp/)
- [Getting started guide](docs/guides/getting-started.md)
- [PyTree workflows](docs/guides/pytree-workflows.md)
- [Warp interop guide](docs/guides/warp.md)
- [API reference map](docs/reference/README.md)

## 🤝 Contributing

Issues and pull requests are welcome, especially around PyTree ergonomics,
Warp integration, and edge cases that show up in real JAX code.

[![PR WELCOME](https://img.shields.io/badge/%F0%9F%A4%AF%20PR%20WELCOME-%E2%86%92-ffcb47?labelColor=black&style=for-the-badge)](https://github.com/liblaf/jarp/pulls)

[![Contributors](https://gh-contributors-gamma.vercel.app/api?repo=liblaf/jarp)](https://github.com/liblaf/jarp/graphs/contributors)

## 🔗 Links

- [Documentation](https://liblaf.github.io/jarp/)
- [Changelog](https://github.com/liblaf/jarp/blob/main/CHANGELOG.md)
- [PyPI](https://pypi.org/project/jarp)
- [Issues](https://github.com/liblaf/jarp/issues)

---

#### 📝 License

Copyright © 2026 [liblaf](https://github.com/liblaf). <br />
This project is [MIT](https://github.com/liblaf/jarp/blob/main/LICENSE)
licensed.
