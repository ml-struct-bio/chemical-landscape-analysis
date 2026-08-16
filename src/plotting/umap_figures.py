"""UMAP scatter figures.

One drawing routine for categorical colourings and one for continuous, both
reading columns the analysis already computed. Nothing here fits, projects, or
derives a value -- it chooses colours and places points.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.common.palette import Palette, colors_for, resolve, to_rgba
from src.plotting.style import raster, save_pdf


DEFAULT_GRAY = "#bfbfbf"

# The panel the previous pipeline's 6_run_pretty_plots_batch_experiment.py drew,
# reproduced deliberately: a 10x10in figure with visible ticks and a full box of
# spines, "UMAP-1"/"UMAP-2" axis labels, and NO title -- a figure's identity came
# from its `<embedding>_<color_by>/` output directory there, and comes from its
# tag directory and filename here, so a title only repeated it. `tight_layout()`
# for the same reason.
FIGSIZE_UMAP = (10.0, 10.0)
POINT_SIZE = 0.25
POINT_ALPHA = 0.25

# Legend markers there were white-stemmed dots at markersize 8 with a frame, and
# labelled with the bare category name -- no point counts.
LEGEND_FONTSIZE = 9
LEGEND_MARKERSIZE = 8

# An uncoloured scatter fell through to a flat steelblue in the old pipeline.
DEFAULT_PLAIN_COLOR = "steelblue"


def _finish(ax) -> None:
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    # style.py drops the top/right spines pipeline-wide; the old UMAP panels
    # carried a full box, so they are put back for these figures only.
    for sp in ax.spines.values():
        sp.set_visible(True)


def _legend_handle(color, label: str):
    return plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
                      markersize=LEGEND_MARKERSIZE, label=label)


def plot_plain(coords: np.ndarray, out_path: Path, *, color: str = DEFAULT_PLAIN_COLOR,
               point_size: float = POINT_SIZE, alpha: float = POINT_ALPHA) -> Path:
    """Every point one colour -- the shape of the map on its own."""
    fig, ax = plt.subplots(figsize=FIGSIZE_UMAP)
    raster(ax.scatter(coords[:, 0], coords[:, 1], s=point_size, alpha=alpha,
                      c=color, linewidths=0))
    _finish(ax)
    fig.tight_layout()
    return save_pdf(fig, out_path)


def _relevant_overrides(overrides: Dict, present: Sequence) -> Dict:
    """Drops palette entries for labels this figure does not contain.

    `palette.resolve` treats naming an absent label as an error, which is the
    right call for catching typos in `--dataset-colors`. But `configs/colors.yaml`
    deliberately names specific Butina cluster ids, and Butina renumbers clusters
    on every re-run -- so a stale entry, or simply drawing a different subset of
    clusters, would make every figure fail. Entries that do not apply are dropped
    with a note instead.
    """
    present_set = set(present)
    keep = {k: v for k, v in (overrides or {}).items() if k in present_set}
    dropped = [k for k in (overrides or {}) if k not in present_set]
    if dropped:
        print(f"  [note] palette entries not present in this figure, ignored: {dropped}")
    return keep


def plot_categorical(coords: np.ndarray, values: np.ndarray, out_path: Path, *,
                     mode: str,
                     palette: Optional[Palette] = None,
                     highlight: Optional[Sequence] = None,
                     other_color: str = DEFAULT_GRAY,
                     point_size: float = POINT_SIZE, alpha: float = POINT_ALPHA,
                     highlight_size: Optional[float] = None,
                     legend: bool = True, max_legend: int = 24) -> Path:
    """Colour by a categorical column, given per-point label values.

    Without `highlight` every point goes down in ONE scatter call carrying a
    per-point colour array, which is what the previous pipeline did and is not a
    cosmetic detail: drawing a separate scatter per label instead paints each
    category wholesale over the ones before it, so at alpha 0.25 whichever source
    sorted last reads as a solid sheet covering the rest. In corpus order the
    sources interleave and the overlap is legible.

    `highlight` restricts colour to the named labels and greys everything else,
    which is how a handful of Butina clusters stay legible against millions of
    background points. Highlighted labels are drawn last, larger and more opaque,
    so they are not buried under the grey.
    """
    palette = palette or Palette()
    present = sorted({v.item() if hasattr(v, "item") else v for v in np.unique(values)})
    nameable = [lv for lv in present if str(lv) != "unknown"]

    if highlight:
        missing = [h for h in highlight if h not in set(present)]
        if missing:
            shown = ", ".join(str(p) for p in present[:40])
            raise SystemExit(
                f"--highlight values not present in this column: {missing}\n"
                f"Available: {shown}" + (" ..." if len(present) > 40 else ""))
        nameable = [lv for lv in nameable if lv in set(highlight)]

    overrides = _relevant_overrides(palette.mode(mode), nameable)
    color_map = resolve(nameable, overrides, mode=mode, ordered_keys=nameable)
    grey = to_rgba(other_color)

    fig, ax = plt.subplots(figsize=FIGSIZE_UMAP)
    is_highlight = bool(highlight)
    handles = []

    if is_highlight:
        bg = ~np.isin(values, nameable)
        if bg.any():
            raster(ax.scatter(coords[bg, 0], coords[bg, 1], s=point_size, alpha=alpha,
                              c=[grey], linewidths=0))

        # Highlighted labels are the one deliberate departure from the old
        # panel, which drew them at the background's size: a handful of 0.25pt
        # clusters among millions of grey points cannot be found by eye.
        hs = highlight_size if highlight_size is not None else point_size * 8
        for lv in nameable:
            sel = values == lv
            if not sel.any():
                continue
            raster(ax.scatter(coords[sel, 0], coords[sel, 1], s=hs,
                              alpha=min(1.0, alpha * 3), c=[color_map[lv]], linewidths=0))
            handles.append(_legend_handle(color_map[lv], str(lv)))
    else:
        # One scatter, points in corpus order -- see the docstring. 'unknown' is
        # a bucket rather than a category, so it takes the background colour.
        full_map = dict(color_map)
        for lv in present:
            full_map.setdefault(lv, grey)
        # Build the per-point RGBA through a lookup table rather than one dict
        # hit per molecule; these arrays run to millions of rows.
        uniq, inverse = np.unique(values, return_inverse=True)
        lut = colors_for([u.item() if hasattr(u, "item") else u for u in uniq], full_map)
        raster(ax.scatter(coords[:, 0], coords[:, 1], c=lut[inverse], s=point_size,
                          alpha=alpha, linewidths=0))

        for lv in nameable:
            handles.append(_legend_handle(color_map[lv], str(lv)))
        if "unknown" in {str(p) for p in present}:
            handles.append(_legend_handle(grey, "unknown"))

    # The old legend carried bare names, so the counts move to stdout rather
    # than being lost outright.
    counts = {str(u): int(c) for u, c in zip(*np.unique(values, return_counts=True))}
    print("  " + ", ".join(f"{k}: {v:,}" for k, v in counts.items()))

    if legend and handles:
        # A legend listing thousands of Butina clusters is not a legend.
        if len(handles) <= max_legend:
            ax.legend(handles=handles, loc="best", frameon=True, fontsize=LEGEND_FONTSIZE)
        else:
            print(f"  [note] {len(handles)} categories -- legend omitted. Use "
                  f"--highlight to name the few you care about.")

    _finish(ax)
    fig.tight_layout()
    return save_pdf(fig, out_path)


def overlay_anchors(ax, anchors, neighbors, *, color_map: Dict[str, tuple],
                    anchor_size: float = 90.0, neighbor_size: float = 22.0,
                    show_neighbors: bool = True) -> List:
    """Draws anchor molecules and their cosine neighbours over an existing map.

    The anchor is a large ringed marker so it stays findable against millions of
    background points; its neighbours are smaller dots in the same hue. Both are
    drawn unrasterized -- there are only a handful, and keeping them vector means
    they stay crisp when the figure is scaled.
    """
    handles = []
    for _, a in anchors.iterrows():
        if a["row"] < 0 or not np.isfinite(a["x"]):
            continue
        color = color_map[a["anchor"]]
        n_drawn = 0
        if show_neighbors and neighbors is not None and len(neighbors):
            nb = neighbors[neighbors["anchor"] == a["anchor"]]
            if len(nb):
                ax.scatter(nb["x"], nb["y"], s=neighbor_size, c=[color], alpha=0.85,
                           linewidths=0.3, edgecolors="white", zorder=8)
                n_drawn = len(nb)
        ax.scatter([a["x"]], [a["y"]], s=anchor_size, c=[color], marker="o",
                   edgecolors="black", linewidths=1.1, zorder=10)
        label = f"{a['anchor']}" + (f" (+{n_drawn} NN)" if n_drawn else "")
        handles.append(plt.Line2D([], [], marker="o", linestyle="", markersize=7,
                                  markeredgecolor="black", color=color, label=label))
    return handles


def _draw_region_zoom(ax, coords: np.ndarray, region, members, color, *,
                      point_size: float, background_color: str = DEFAULT_GRAY) -> None:
    """A zoomed view of one region: local background plus its molecules."""
    cx, cy, r = region["center_x"], region["center_y"], region["radius"]
    lo_x, hi_x, lo_y, hi_y = cx - r, cx + r, cy - r, cy + r
    near = ((coords[:, 0] >= lo_x) & (coords[:, 0] <= hi_x) &
            (coords[:, 1] >= lo_y) & (coords[:, 1] <= hi_y))
    if near.any():
        raster(ax.scatter(coords[near, 0], coords[near, 1], s=max(point_size, 2.0),
                          alpha=0.45, c=background_color, edgecolors="none"))
    if members is not None and len(members):
        ax.scatter(members["x"], members["y"], s=26, c=[color], alpha=0.95,
                   linewidths=0.3, edgecolors="white", zorder=6)
    ax.set_xlim(lo_x, hi_x)
    ax.set_ylim(lo_y, hi_y)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(region["label"], fontsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor(color)
        sp.set_linewidth(1.4)


def plot_with_insets(coords: np.ndarray, out_path: Path, *,
                     regions, region_mols, color_map: Dict[str, tuple],
                     anchors=None, neighbors=None,
                     base_color: str = DEFAULT_GRAY, point_size: float = POINT_SIZE,
                     alpha: float = POINT_ALPHA, mols_per_row: int = 3,
                     sub_img_size: tuple = (170, 170), max_insets: int = 6) -> Path:
    """The main map beside a column of insets, one per region.

    Each inset row is [zoomed scatter | structure grid], so a region's location
    and what actually lives there are read together. Regions past `max_insets`
    are dropped with a note rather than squeezed into an unreadable strip.
    """
    from src.plotting.mol_render import render_mol_grid

    shown = regions.head(max_insets)
    if len(regions) > len(shown):
        print(f"  [note] {len(regions)} regions, drawing the first {len(shown)}. "
              f"Raise --max-insets to include more.")
    n = max(len(shown), 1)

    fig = plt.figure(figsize=(7.0 + 4.6, max(5.0, 1.75 * n)))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.12)
    ax_main = fig.add_subplot(outer[0, 0])
    raster(ax_main.scatter(coords[:, 0], coords[:, 1], s=point_size, alpha=alpha,
                           c=base_color, linewidths=0))

    inner = outer[0, 1].subgridspec(n, 2, width_ratios=[1.0, 1.25], hspace=0.45, wspace=0.05)

    for i, (_, region) in enumerate(shown.iterrows()):
        color = color_map.get(region["label"], color_map.get(region["region_id"], "#d62728"))
        members = (region_mols[region_mols["region_id"] == region["region_id"]]
                   .sort_values("rank") if region_mols is not None and len(region_mols)
                   else None)

        # Mark the region on the main map, so an inset can be located.
        r = region["radius"]
        ax_main.add_patch(plt.Rectangle((region["center_x"] - r, region["center_y"] - r),
                                        2 * r, 2 * r, fill=False, edgecolor=color,
                                        linewidth=1.2, zorder=9))

        _draw_region_zoom(fig.add_subplot(inner[i, 0]), coords, region, members, color,
                          point_size=point_size)

        ax_grid = fig.add_subplot(inner[i, 1])
        smis = members["smiles"].tolist() if members is not None else []
        if smis:
            ax_grid.imshow(np.asarray(render_mol_grid(smis, mols_per_row=mols_per_row,
                                                      sub_img_size=sub_img_size)))
        else:
            ax_grid.text(0.5, 0.5, "no molecules", ha="center", va="center",
                         fontsize=7, color="0.5", transform=ax_grid.transAxes)
        ax_grid.axis("off")

    handles = []
    if anchors is not None and len(anchors):
        handles = overlay_anchors(ax_main, anchors, neighbors, color_map=color_map)
    if handles:
        ax_main.legend(handles=handles, loc="best", frameon=True, fontsize=LEGEND_FONTSIZE)

    _finish(ax_main)
    return save_pdf(fig, out_path)


def plot_anchors(coords: np.ndarray, out_path: Path, *, anchors, neighbors,
                 color_map: Dict[str, tuple], base_color: str = DEFAULT_GRAY,
                 point_size: float = POINT_SIZE, alpha: float = POINT_ALPHA,
                 show_neighbors: bool = True) -> Path:
    """The map with anchor molecules (and optionally their neighbours) marked."""
    fig, ax = plt.subplots(figsize=FIGSIZE_UMAP)
    raster(ax.scatter(coords[:, 0], coords[:, 1], s=point_size, alpha=alpha,
                      c=base_color, linewidths=0))
    handles = overlay_anchors(ax, anchors, neighbors, color_map=color_map,
                              show_neighbors=show_neighbors)
    if handles:
        ax.legend(handles=handles, loc="best", frameon=True, fontsize=LEGEND_FONTSIZE)
    _finish(ax)
    fig.tight_layout()
    return save_pdf(fig, out_path)


def plot_neighbor_similarity(neighbors, out_path: Path, *, title: str,
                             color_map: Dict[str, tuple]) -> Path:
    """Cosine similarity vs. neighbour rank, one line per anchor.

    How fast this falls away says whether an anchor sits in a tight, well-defined
    neighbourhood or in a diffuse part of the space -- which the map alone cannot
    show, since UMAP distorts density.
    """
    fig, ax = plt.subplots(figsize=(5.0, 3.2), constrained_layout=True)
    for name, group in neighbors.groupby("anchor"):
        group = group.sort_values("rank")
        ax.plot(group["rank"] + 1, group["cosine_sim"], marker="o", markersize=3,
                linewidth=1.2, color=color_map.get(name), label=str(name))
    ax.set_xlabel("neighbour rank")
    ax.set_ylabel("cosine similarity")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    return save_pdf(fig, out_path)


def plot_continuous(coords: np.ndarray, values: np.ndarray, out_path: Path, *,
                    label: str, cmap: str = "viridis",
                    point_size: float = POINT_SIZE, alpha: float = POINT_ALPHA,
                    pct_lo: float = 1.0, pct_hi: float = 99.0) -> Path:
    """Colour by a continuous column, clipped to a robust percentile range.

    Molecular and spectral properties have long tails -- one 10,000 Da outlier
    against true min/max would put every ordinary molecule in the bottom 5% of
    the colormap and render the figure a single colour. NaNs (unparseable SMILES,
    unmatched spectra) are drawn grey rather than dropped, so the map keeps its
    shape and missingness stays visible.
    """
    finite = np.isfinite(values)
    fig, ax = plt.subplots(figsize=FIGSIZE_UMAP)

    if (~finite).any():
        raster(ax.scatter(coords[~finite, 0], coords[~finite, 1], s=point_size,
                          alpha=alpha * 0.6, c=DEFAULT_GRAY, linewidths=0))

    if finite.any():
        vmin, vmax = np.percentile(values[finite], [pct_lo, pct_hi])
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = float(np.min(values[finite])), float(np.max(values[finite]))
            if vmin == vmax:
                vmax = vmin + 1e-9
        sc = ax.scatter(coords[finite, 0], coords[finite, 1], s=point_size, alpha=alpha,
                        c=values[finite], cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0)
        raster(sc)
        cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
        cbar.set_label(label, fontsize=8)
        cbar.ax.tick_params(labelsize=7)
        cbar.solids.set_alpha(1.0)

    # The old panel had no title to hang this off, so it is reported instead of
    # drawn -- missingness still has to be visible somewhere.
    n_missing = int((~finite).sum())
    if n_missing:
        print(f"  [note] {label}: {n_missing:,} missing value(s), drawn grey.")

    _finish(ax)
    fig.tight_layout()
    return save_pdf(fig, out_path)
