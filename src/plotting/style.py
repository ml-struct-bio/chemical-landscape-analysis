"""One place for figure style, and the PDF rules the whole pipeline follows.

The previous pipeline had no shared style at all -- each plotting module carried
its own `DPI` constant (300 in some, 600 in others, 900 in one), so figures from
different scripts did not compose into a coherent set.

Output is PDF. Two consequences worth being deliberate about:

* **Fonts.** `pdf.fonttype = 42` (and `ps.fonttype = 42`) embeds TrueType rather
  than shipping Type-3 outlines, so text stays selectable and editable in
  Illustrator/Inkscape. `font.family = "Arial"` pins the typeface itself, so
  every figure in the pipeline sets type in the same font regardless of which
  script drew it.
* **Dense point layers.** A UMAP scatter of ~2.5M molecules stored as vector
  paths produces a PDF hundreds of MB large that no viewer will open. Pass
  `rasterized=True` on those artists (`raster()` below does it for you) and the
  points are flattened to a `RASTER_DPI` image while axes, ticks, labels and
  legends stay vector. This is the single most important habit in this module.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# Resolution used for rasterized layers inside an otherwise-vector PDF.
RASTER_DPI = 600

# Shared figure sizes (inches). Widths chosen against a single- and
# double-column journal page so figures need no rescaling on import.
FIGSIZE_SINGLE = (3.4, 3.0)
FIGSIZE_DOUBLE = (7.0, 3.0)
FIGSIZE_SQUARE = (5.0, 5.0)
FIGSIZE_WIDE = (10.0, 4.0)

RCPARAMS = {
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "Arial",
    "svg.fonttype": "none",
    "savefig.format": "pdf",
    "savefig.bbox": "tight",
    "savefig.transparent": False,
    "figure.dpi": 150,
    "savefig.dpi": RASTER_DPI,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "figure.autolayout": False,
}


def apply_style() -> None:
    """Call once at the top of every plotting entrypoint's `main()`."""
    plt.rcParams.update(RCPARAMS)


def raster(artist: Any) -> Any:
    """Marks an artist as rasterized and returns it, so dense layers can be
    written inline:  `raster(ax.scatter(x, y, s=1))`."""
    try:
        artist.set_rasterized(True)
    except AttributeError:
        pass
    return artist


def save_pdf(fig: Optional[plt.Figure], out_path: Path) -> Path:
    """Saves as PDF (forcing the suffix) and closes the figure.

    Closing matters here: plotting scripts routinely emit dozens of figures in
    one process, and matplotlib keeps every un-closed one alive.
    """
    fig = fig if fig is not None else plt.gcf()
    out_path = Path(out_path).with_suffix(".pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", dpi=RASTER_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path
