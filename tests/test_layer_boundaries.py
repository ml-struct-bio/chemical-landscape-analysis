"""Enforces the analysis/plotting separation by scanning imports.

The previous pipeline had the same two-layer design as a documented convention
and nothing else, and it eroded: four analysis modules imported matplotlib,
seven plotting modules imported from `src/analysis/` (one of them moving
similarity matrices to a GPU at draw time), and one analysis module imported a
plotting module outright. These tests are the mechanism that was missing.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterator, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]

# Drawing libraries. An analysis that reaches for one of these is deciding how a
# figure looks, which is the plotting script's job -- including the tempting
# small case of building a colormap or picking tab10 colors to store in its
# output. Those belong in src/common/palette.py, which both layers may use.
DRAWING_MODULES = {"matplotlib", "seaborn", "PIL", "plotly"}

# Heavy compute. A plotting module importing one of these is re-deriving
# something the analysis should have computed and written to data/.
COMPUTE_MODULES = {"sklearn", "umap", "torch", "hydra"}


def _module_roots(tree: ast.AST) -> Iterator[Tuple[str, int]]:
    """Yields (dotted module name, lineno) for every import in the file."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, node.lineno
        elif isinstance(node, ast.ImportFrom):
            # Relative imports have no module path to police.
            if node.level == 0 and node.module:
                yield node.module, node.lineno


def _violations(path: Path, banned: set[str], banned_prefixes: tuple[str, ...] = ()) -> List[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    found = []
    for name, lineno in _module_roots(tree):
        root = name.split(".")[0]
        if root in banned or any(name == p or name.startswith(p + ".") for p in banned_prefixes):
            found.append(f"{path.relative_to(REPO_ROOT)}:{lineno} imports {name}")
    return found


def _py_files(*relative_dirs: str) -> List[Path]:
    files: List[Path] = []
    for rel in relative_dirs:
        directory = REPO_ROOT / rel
        if directory.exists():
            files.extend(sorted(p for p in directory.rglob("*.py")
                                if "__pycache__" not in p.parts))
    return files


def test_analysis_library_does_not_draw():
    """`src/analysis/` computes; it must not import a drawing library."""
    bad = [v for p in _py_files("src/analysis") for v in _violations(p, DRAWING_MODULES)]
    assert not bad, "analysis modules must not import drawing libraries:\n" + "\n".join(bad)


def test_plotting_library_does_not_compute():
    """`src/plotting/` draws what is already on disk. Importing sklearn/umap/
    torch means it is recomputing; importing `src.analysis` means the two layers
    are coupled again and a plot script cannot run on its own."""
    bad = [v for p in _py_files("src/plotting")
           for v in _violations(p, COMPUTE_MODULES, banned_prefixes=("src.analysis",))]
    assert not bad, "plotting modules must not compute or import src.analysis:\n" + "\n".join(bad)


def test_entrypoints_do_not_cross_layers():
    """An `analysis/` script that imports `src.plotting` (or the reverse) has
    re-merged the two programs no matter what the libraries look like."""
    bad = [v for p in _py_files("analysis") for v in _violations(p, set(), ("src.plotting",))]
    bad += [v for p in _py_files("plotting") for v in _violations(p, set(), ("src.analysis",))]
    assert not bad, "analysis/ and plotting/ entrypoints must not import each other's layer:\n" + "\n".join(bad)


def test_plotting_never_reads_the_cache():
    """`cache/` holds pickled refit objects and is analysis-only. A plotting
    script reaching in has reintroduced exactly the coupling this layout
    removes -- if a figure needs something from a fit, the analysis must
    evaluate it and write the answer into `data/`."""
    bad = []
    for path in _py_files("plotting", "src/plotting"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == "cache_dir":
                bad.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno} uses cache_dir()")
    assert not bad, "plotting must not touch cache/:\n" + "\n".join(bad)


def test_src_has_no_init_files():
    """`src/` must stay a PEP 420 namespace package.

    `src/extraction/nmr3d.py`'s import scope works by evicting the cached `src`
    module so Python re-resolves it against `--nmr3d-root`, whose own `src/` is
    a *regular* package. Adding an `__init__.py` here would make both regular
    and break that resolution -- with a `ModuleNotFoundError: No module named
    'src.model'` that points nowhere near the cause."""
    inits = [p for p in (REPO_ROOT / "src").rglob("__init__.py")]
    assert not inits, ("src/ must remain a namespace package (no __init__.py):\n"
                       + "\n".join(str(p.relative_to(REPO_ROOT)) for p in inits))
