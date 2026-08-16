"""Where every pipeline artifact lives.

The whole layout is four sibling trees at the repo root:

    analysis/NN_slug.py   ->  data/NN_slug/<tag>/     (figure data)
    plotting/NN_slug.py   <-  data/NN_slug/<tag>/
                          ->  figures/NN_slug/<tag>/  (PDFs)
                              cache/...               (refit objects)

Paths here are **deterministic**: `data_dir("04_pca", "main")` is always the
same directory, and re-running an analysis overwrites it in place.

That is a deliberate break from the previous pipeline, whose
`resolve_experiment_dir()` appended a `_1`, `_2`, ... suffix whenever the target
name already existed. That looked like an anti-clobber guarantee, but because
every plotting path derived its cache location from the freshly-created (empty)
directory, a second run could never find the artifacts the first one wrote --
which is why several `experiments/*/analysis/` directories in the old repo hold
nothing but a manifest. Overwriting is safe here for a different reason: `data/`
is derived, and each directory's `manifest.json` records exactly the inputs and
parameters needed to rebuild it. Use `--tag` to keep variants side by side.
"""
from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = REPO_ROOT / "data"
FIGURES_ROOT = REPO_ROOT / "figures"
CACHE_ROOT = REPO_ROOT / "cache"

DEFAULT_TAG = "main"


def _leaf(root: Path, slug: str, tag: str, create: bool) -> Path:
    if not slug:
        raise ValueError("slug must be a non-empty analysis name, e.g. '04_pca'")
    if not tag:
        raise ValueError("tag must be non-empty (default 'main')")
    for part, name in ((slug, "slug"), (tag, "tag")):
        if "/" in part or part in (".", ".."):
            raise ValueError(f"{name} must be a single path component, got {part!r}")
    path = root / slug / tag
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def data_dir(slug: str, tag: str = DEFAULT_TAG, *, create: bool = False) -> Path:
    """`data/<slug>/<tag>/` -- the figure data an analysis writes and its
    plotting counterpart reads. Pass `create=True` from the analysis side."""
    return _leaf(DATA_ROOT, slug, tag, create)


def figures_dir(slug: str, tag: str = DEFAULT_TAG, *, create: bool = False) -> Path:
    """`figures/<slug>/<tag>/` -- PDFs. Only plotting scripts write here."""
    return _leaf(FIGURES_ROOT, slug, tag, create)


def cache_dir(name: str, *, create: bool = False) -> Path:
    """`cache/<name>/` -- heavyweight refit objects (UMAP fits, fitted scalers)
    that are NOT part of the analysis->plotting contract.

    This is the one place pickles are allowed, and it is deliberately walled
    off: **only `analysis/` scripts may read or write it.** A plotting script
    that reaches in here has reintroduced exactly the coupling this layout
    exists to remove -- if a figure needs something from a fit, the analysis
    must evaluate it and write the answer into `data/`."""
    if not name:
        raise ValueError("cache name must be non-empty")
    path = CACHE_ROOT / name
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path
