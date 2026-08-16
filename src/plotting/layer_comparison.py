"""Plots for the decoder layer-comparison analysis.

Port of the previous pipeline's `src/plotting/layer_comparison_plots.py`: the
three layer-sweep summary plots (property R^2, cluster quality, ECFP-vs-
embedding NN overlap), a PC-interpretability-vs-depth sweep, and PC-traversal
filmstrips for the representative decoder layers `src/analysis/layer_comparison.py`
selected. No property-direction traversal and no per-representation
PC1/PC2-vs-descriptor scatter grid -- see that module's docstring for why.

Consumes `data/07_layer_comparison/<tag>/` only.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.plotting.mol_render import mol_to_image
from src.plotting.style import raster, save_pdf

QUALITATIVE_PALETTE = plt.cm.tab10(np.linspace(0, 1, 10))

# Decoder representations are named `decoder_<x|y>_L<layer>_t<timestep>` by
# EmbeddingSpec.slug; the baselines (ECFP, global_cond) don't match and have
# no timestep.
_DECODER_NAME_RE = re.compile(r"^decoder_[xy]_L(-?\d+)_t(.+)$")


def _style_axis(ax, title: str) -> None:
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.axvline(-0.75, color="gray", linestyle=":", linewidth=1)
    ax.set_title(title, fontsize=11, fontweight="bold")


def _parse_timestep(name) -> Optional[float]:
    match = _DECODER_NAME_RE.match(str(name))
    if match is None:
        return None
    try:
        return float(match.group(2))
    except ValueError:
        return None


def timestep_facets(df: pd.DataFrame) -> List[Tuple[Optional[float], pd.DataFrame]]:
    """Split a layer-sweep results frame into one sub-frame per decoder timestep.

    `representation` encodes both axes (`decoder_<stream>_L<layer>_t<timestep>`)
    but `x_pos` carries only the layer, so every timestep otherwise lands on the
    same x and the sweep lines connect across unrelated timesteps. Rows with no
    timestep (the ECFP/global_cond baselines) are repeated into every facet so
    each panel stays self-contained, and each facet is sorted by `x_pos` so its
    polyline runs monotonically with depth.

    Returns [(timestep, frame)]; a single [(None, df)] facet when nothing
    parses (e.g. a baselines-only frame).
    """
    timesteps = df["representation"].map(_parse_timestep)
    baselines = df[timesteps.isna()]
    unique_ts = sorted(timesteps.dropna().unique())
    if not unique_ts:
        return [(None, df.sort_values("x_pos"))]
    return [
        (ts, pd.concat([baselines, df[timesteps == ts]], ignore_index=True).sort_values("x_pos"))
        for ts in unique_ts
    ]


def _facet_title(timestep: Optional[float]) -> str:
    return "" if timestep is None else f"timestep = {timestep:g}"


def _set_layer_xaxis(ax, sub: pd.DataFrame) -> None:
    """Tick every representation explicitly -- decoder layers by their integer
    depth, baselines by name (their x_pos values are arbitrary negative
    sentinels and read as meaningless coordinates)."""
    ticks, labels = [], []
    for _, row in sub.sort_values("x_pos").iterrows():
        ticks.append(row["x_pos"])
        labels.append(str(row["representation"]) if _parse_timestep(row["representation"]) is None
                      else f"{int(row['x_pos'])}")
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")


# -----------------------------------------------------------------------------
# Layer-sweep summary plots
# -----------------------------------------------------------------------------


def plot_property_r2_vs_layer(df: pd.DataFrame, property_names: List[str], out_path: Path) -> Path:
    """One panel per decoder timestep (baselines repeated in each), sharing a
    y-axis so depth trends are directly comparable across timesteps."""
    facets = timestep_facets(df)
    ncols = len(facets)
    fig, axes = plt.subplots(1, ncols, figsize=(max(6.5, 5.2 * ncols), 5.4),
                             squeeze=False, sharey=True)
    for ax_i, (ax, (timestep, sub)) in enumerate(zip(axes[0], facets)):
        for i, prop in enumerate(property_names):
            col = f"r2_{prop}"
            if col not in sub.columns:
                continue
            ax.plot(sub["x_pos"], sub[col], color=QUALITATIVE_PALETTE[i % 10],
                    alpha=0.5, linewidth=1.2, zorder=1)
            ax.scatter(sub["x_pos"], sub[col], color=QUALITATIVE_PALETTE[i % 10], s=60,
                       edgecolors="black", linewidths=0.8, label=prop, zorder=2)
        ax.set_xlabel("Decoder layer (baselines left of dotted line)")
        if ax_i == 0:
            ax.set_ylabel("held-out R²")
            ax.legend(fontsize=8, loc="best", framealpha=0.9)
        _style_axis(ax, _facet_title(timestep))
        _set_layer_xaxis(ax, sub)
    fig.suptitle("Property linear-decodability vs. layer depth  ↑ Higher is better",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return save_pdf(fig, out_path)


def plot_cluster_quality_vs_layer(df: pd.DataFrame, out_path: Path, include_silhouette: bool = False) -> Path:
    """Grid of timestep (rows) x cluster-quality metric (columns); each column
    shares a y-axis so a metric is comparable across timesteps."""
    metrics = ["calinski_harabasz", "davies_bouldin", "knn_purity"] + (["silhouette"] if include_silhouette else [])
    directions = {"calinski_harabasz": "↑ Higher is better", "davies_bouldin": "↓ Lower is better",
                  "knn_purity": "↑ Higher is better", "silhouette": "↑ Higher is better"}
    facets = timestep_facets(df)
    nrows, ncols = len(facets), len(metrics)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 4.2 * nrows),
                             squeeze=False, sharey="col")
    for row_i, (timestep, sub) in enumerate(facets):
        for ax, metric in zip(axes[row_i], metrics):
            ax.plot(sub["x_pos"], sub[metric], color=QUALITATIVE_PALETTE[0],
                    alpha=0.5, linewidth=1.2, zorder=1)
            ax.scatter(sub["x_pos"], sub[metric], color=QUALITATIVE_PALETTE[0], s=70,
                       edgecolors="black", linewidths=0.8, zorder=2)
            ax.set_xlabel("Decoder layer")
            ax.set_ylabel(metric.replace("_", " ").title())
            title = f"{metric.replace('_', ' ').title()}\n{directions[metric]}"
            _style_axis(ax, f"{_facet_title(timestep)}\n{title}" if timestep is not None else title)
            _set_layer_xaxis(ax, sub)
    fig.tight_layout()
    return save_pdf(fig, out_path)


def plot_nn_overlap_vs_layer(df: pd.DataFrame, k_values: List[int], out_path: Path) -> Path:
    """Grid of timestep (rows) x {Tanimoto-cosine correlation, top-k neighbor
    overlap} (columns); each column shares a y-axis across timesteps."""
    facets = timestep_facets(df)
    nrows, ncols = len(facets), 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.2 * nrows),
                             squeeze=False, sharey="col")
    for row_i, (timestep, sub) in enumerate(facets):
        prefix = f"{_facet_title(timestep)}\n" if timestep is not None else ""

        ax = axes[row_i][0]
        for name, color in [("tanimoto_cosine_pearson", QUALITATIVE_PALETTE[0]),
                             ("tanimoto_cosine_spearman", QUALITATIVE_PALETTE[1])]:
            ax.plot(sub["x_pos"], sub[name], color=color, alpha=0.5, linewidth=1.2, zorder=1)
            ax.scatter(sub["x_pos"], sub[name], color=color, s=60, edgecolors="black",
                       linewidths=0.8, label=name.split("_")[-1], zorder=2)
        ax.set_xlabel("Decoder layer")
        ax.set_ylabel("correlation")
        _style_axis(ax, f"{prefix}ECFP-Tanimoto vs. embedding-cosine\n↑ Higher = more structure-faithful")
        _set_layer_xaxis(ax, sub)
        ax.legend(fontsize=8)

        ax = axes[row_i][1]
        for i, k in enumerate(k_values):
            col = f"top{k}_overlap"
            if col not in sub.columns:
                continue
            ax.plot(sub["x_pos"], sub[col], color=QUALITATIVE_PALETTE[i % 10],
                    alpha=0.5, linewidth=1.2, zorder=1)
            ax.scatter(sub["x_pos"], sub[col], color=QUALITATIVE_PALETTE[i % 10], s=60,
                       edgecolors="black", linewidths=0.8, label=f"top-{k}", zorder=2)
        ax.set_xlabel("Decoder layer")
        ax.set_ylabel("neighbor-set overlap")
        _style_axis(ax, f"{prefix}ECFP vs. embedding top-k neighbor overlap\n↑ Higher = more structure-faithful retrieval")
        _set_layer_xaxis(ax, sub)
        ax.legend(fontsize=8)

    fig.tight_layout()
    return save_pdf(fig, out_path)


# -----------------------------------------------------------------------------
# PC1/PC2 <-> RDKit-descriptor interpretability, swept across decoder depth
# -----------------------------------------------------------------------------


def plot_pc_interpretability_vs_layer(pc_best_df: pd.DataFrame, out_path: Path) -> Path:
    """`pc_best_df`: long-format (representation, x_pos, pc, best_descriptor, r)
    from `pc_best_descriptor.csv`, `pc` in {'PC1', 'PC2'}.

    Grid of timestep (rows) x {PC1, PC2} (columns): each representation's |r|
    against its own best-correlated RDKit descriptor, swept across decoder
    depth (baselines left of the dotted line) -- the same layer-sweep style as
    plot_property_r2_vs_layer, but for unsupervised PC interpretability instead
    of supervised property decodability.
    """
    rows = []
    for (rep, x_pos), group in pc_best_df.groupby(["representation", "x_pos"], sort=False):
        row = {"representation": rep, "x_pos": x_pos}
        for _, r in group.iterrows():
            r_val = r["r"]
            row[f"{r['pc']}_abs_r"] = abs(r_val) if pd.notna(r_val) else np.nan
            row[f"{r['pc']}_descriptor"] = r["best_descriptor"]
        rows.append(row)
    df = pd.DataFrame(rows)

    facets = timestep_facets(df)
    nrows = len(facets)
    fig, axes = plt.subplots(nrows, 2, figsize=(11, 4.6 * nrows), squeeze=False)
    for row_i, (timestep, sub) in enumerate(facets):
        prefix = f"{_facet_title(timestep)}\n" if timestep is not None else ""
        for pc_i, pc_label in enumerate(["PC1", "PC2"]):
            ax = axes[row_i][pc_i]
            if f"{pc_label}_abs_r" not in sub.columns:
                ax.axis("off")
                continue
            xs = sub["x_pos"].to_numpy(dtype=float)
            ys = sub[f"{pc_label}_abs_r"].to_numpy(dtype=float)
            ax.plot(xs, ys, color=QUALITATIVE_PALETTE[pc_i], alpha=0.5, linewidth=1.2, zorder=1)
            ax.scatter(xs, ys, color=QUALITATIVE_PALETTE[pc_i], s=60,
                       edgecolors="black", linewidths=0.8, zorder=2)
            for x, y, lab in zip(xs, ys, sub[f"{pc_label}_descriptor"]):
                if lab and isinstance(lab, str) and np.isfinite(y):
                    ax.annotate(lab, (x, y), textcoords="offset points", xytext=(0, 6),
                                fontsize=6, ha="center", rotation=45)
            ax.set_xlabel("Decoder layer (baselines at left of dotted line)")
            ax.set_ylabel("|r| of best-correlated RDKit descriptor")
            ax.set_ylim(0, 1.05)
            _style_axis(ax, f"{prefix}{pc_label} interpretability vs. layer depth\n"
                            "↑ Higher = more property-interpretable")
            _set_layer_xaxis(ax, sub)
    fig.tight_layout()
    return save_pdf(fig, out_path)


# -----------------------------------------------------------------------------
# PC-traversal filmstrips (structure only -- no spectral panel, this analysis
# has no spectral features wired in). Reuses mol_render.mol_to_image; the step
# resolution itself (real molecules, percentile range) already happened in
# src/analysis/layer_comparison.py via geometry.traversal_steps.
# -----------------------------------------------------------------------------


def pc_traversal_plot(trav_df: pd.DataFrame, background: np.ndarray, pc: int, other_dim: int,
                      representation: str, out_path: Path, mol_size: int = 220,
                      pct_lo: float = 1.0, pct_hi: float = 99.0) -> Path:
    sub = trav_df[trav_df["pc"] == pc].sort_values("step")
    n_steps = len(sub)
    fig, axes = plt.subplots(2, n_steps, figsize=(n_steps * 2, 4))
    if n_steps == 1:
        axes = axes.reshape(2, 1)

    bx, by = background[:, pc - 1], background[:, other_dim - 1]
    for i, (_, r) in enumerate(sub.iterrows()):
        img = mol_to_image(r["smiles"], size=mol_size)
        if img is not None:
            axes[0, i].imshow(img)
        axes[0, i].axis("off")
        axes[0, i].set_title(f"PC{pc}={r['pc_value']:.2f}", fontsize=6)

        raster(axes[1, i].scatter(bx, by, s=0.5, alpha=0.2, c="lightgray", edgecolors="none"))
        axes[1, i].scatter([r["x"]], [r["y"]], c="red", s=30, zorder=5)
        axes[1, i].set_xlabel(f"PC{pc}", fontsize=6)
        axes[1, i].set_ylabel(f"PC{other_dim}", fontsize=6)
        axes[1, i].tick_params(labelsize=5)
        for sp in ("top", "right"):
            axes[1, i].spines[sp].set_visible(False)

    fig.suptitle(f"{representation}: PC{pc} traversal (all other PCs fixed at median; "
                f"range = [{pct_lo:.0f}th, {pct_hi:.0f}th] percentile)", fontsize=10)
    fig.tight_layout()
    return save_pdf(fig, out_path)


# -----------------------------------------------------------------------------
# Orchestrator: renders the full output set from data/07_layer_comparison/<tag>/
# -----------------------------------------------------------------------------


def plot_layer_comparison(data_dir: Path, out_dir: Path, *, params: dict) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}

    results_df = pd.read_csv(data_dir / "metrics.csv")
    property_names = params.get("property_names", [])
    include_silhouette = params.get("include_silhouette", False)
    k_neighbors = params.get("k_neighbors", [5, 10, 25])

    written["property_r2"] = plot_property_r2_vs_layer(
        results_df, property_names, out_dir / "property_r2_vs_layer.pdf")

    if all(m in results_df.columns for m in ("calinski_harabasz", "davies_bouldin", "knn_purity")):
        written["cluster_quality"] = plot_cluster_quality_vs_layer(
            results_df, out_dir / "cluster_quality_vs_layer.pdf", include_silhouette)

    if "tanimoto_cosine_pearson" in results_df.columns:
        written["nn_overlap"] = plot_nn_overlap_vs_layer(
            results_df, k_neighbors, out_dir / "nn_overlap_vs_layer.pdf")

    pc_best_path = data_dir / "pc_best_descriptor.csv"
    if pc_best_path.exists():
        pc_best_df = pd.read_csv(pc_best_path)
        if not pc_best_df.empty:
            written["pc_interpretability"] = plot_pc_interpretability_vs_layer(
                pc_best_df, out_dir / "pc12_interpretability_vs_layer.pdf")

    trav_path = data_dir / "traversal.csv"
    stats_path = data_dir / "traversal_stats.csv"
    if trav_path.exists() and stats_path.exists():
        trav_df = pd.read_csv(trav_path)
        stats_df = pd.read_csv(stats_path)
        pct_lo = params.get("pc_pct_lo", 1.0)
        pct_hi = params.get("pc_pct_hi", 99.0)
        for representation in params.get("traversal_representations", []):
            bg_path = data_dir / f"traversal_background_{representation}.npz"
            if not bg_path.exists():
                print(f"[warn] No {bg_path.name} -- skipping traversal filmstrips for {representation}.")
                continue
            background = np.load(bg_path)["coords"]
            rep_trav = trav_df[trav_df["representation"] == representation]
            rep_stats = stats_df[stats_df["representation"] == representation]
            for _, stat_row in rep_stats.iterrows():
                pc = int(stat_row["pc"])
                key = f"{representation}_pc{pc}"
                written[key] = pc_traversal_plot(
                    rep_trav, background, pc, int(stat_row["other_dim"]), representation,
                    out_dir / f"{representation}_pc{pc}_traversal.pdf",
                    pct_lo=pct_lo, pct_hi=pct_hi)

    return written
