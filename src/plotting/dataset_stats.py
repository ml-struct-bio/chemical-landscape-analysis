"""Dataset-statistics figures. Molecular half only -- see
`src/analysis/dataset_stats.py` for what each table actually holds.

Every function takes a `palette: Dict[str, tuple]` (dataset label -> RGBA)
resolved by the caller through `src.common.palette`, so a dataset's color stays
consistent with every other figure in the pipeline instead of an independent
tab10-by-sorted-index assignment.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.plotting.style import save_pdf

# Properties whose distributions span orders of magnitude and are unreadable
# on a linear count axis. Not currently consumed (kept for parity with the
# previous pipeline / future log-scale support).
LOG_Y_FEATURES = {"BertzCT", "MolWt", "HeavyAtomMolWt", "ExactMolWt", "LabuteASA"}


def _grid(n: int, ncols: int = 3, panel: tuple = (4.0, 3.0)):
    """Returns the axes as a LIST, not `axes.flat`. `.flat` is a one-shot
    iterator: zipping it against the feature list consumes it, after which
    indexing for the legend raises IndexError and the trailing blank-panel
    cleanup silently does nothing."""
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel[0] * ncols, panel[1] * nrows),
                              squeeze=False)
    return fig, list(axes.flat), nrows * ncols


def _finish(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _values(stats_df: pd.DataFrame, ds: str, feat: str) -> np.ndarray:
    return stats_df.loc[stats_df["dataset"] == ds, feat].dropna().to_numpy()


# -----------------------------------------------------------------------------
# Counts / sizes
# -----------------------------------------------------------------------------

def plot_dataset_counts(counts_df: pd.DataFrame, out_path: Path) -> Path:
    """Per-dataset totals, stacked by split, with exact counts annotated."""
    pivot = counts_df.pivot(index="dataset", columns="split", values="n_molecules").fillna(0)
    order = [s for s in ("train", "val", "test") if s in pivot.columns]
    pivot = pivot[order] if order else pivot

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    bottom = np.zeros(len(pivot))
    colors = plt.cm.Blues(np.linspace(0.45, 0.85, max(len(pivot.columns), 1)))
    x = np.arange(len(pivot))
    for i, split in enumerate(pivot.columns):
        ax.bar(x, pivot[split].to_numpy(), bottom=bottom, label=split, color=colors[i])
        bottom += pivot[split].to_numpy()
    for xi, total in zip(x, bottom):
        ax.text(xi, total, f"{int(total):,}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=20, ha="right")
    ax.set_ylabel("n molecules")
    ax.set_title("Dataset sizes by split")
    ax.legend(frameon=False, fontsize=8)
    ax.margins(y=0.12)
    _finish(ax)
    return save_pdf(fig, out_path)


# -----------------------------------------------------------------------------
# Descriptor distributions
# -----------------------------------------------------------------------------

def plot_descriptor_boxplots(stats_df: pd.DataFrame, features: Sequence[str], out_path: Path,
                              palette: Dict[str, tuple]) -> Path:
    datasets = sorted(stats_df["dataset"].unique())
    fig, axes, n_slots = _grid(len(features))

    for ax, feat in zip(axes, features):
        data, labels = [], []
        for ds in datasets:
            vals = _values(stats_df, ds, feat)
            if len(vals):
                data.append(vals)
                labels.append(ds)
        if not data:
            ax.axis("off")
            continue
        bp = ax.boxplot(data, patch_artist=True, tick_labels=labels, showfliers=False)
        for box, ds in zip(bp["boxes"], labels):
            box.set(facecolor=palette[ds], alpha=0.75)
        for median in bp["medians"]:
            median.set(color="black", linewidth=1.2)
        ax.set_title(feat, fontsize=10)
        ax.tick_params(axis="x", labelrotation=20, labelsize=8)
        ax.grid(True, axis="y", alpha=0.2)
        _finish(ax)

    for ax in list(axes)[len(features):]:
        ax.axis("off")
    fig.suptitle("Descriptor distributions by dataset (outliers hidden)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return save_pdf(fig, out_path)


def plot_descriptor_histograms(stats_df: pd.DataFrame, features: Sequence[str], out_path: Path,
                                palette: Dict[str, tuple]) -> Path:
    """Shared bins across datasets and density normalization, so the figure
    compares chemistry rather than dataset size."""
    datasets = sorted(stats_df["dataset"].unique())
    fig, axes, _ = _grid(len(features))

    for ax, feat in zip(axes, features):
        pooled = stats_df[feat].dropna().to_numpy()
        if len(pooled) == 0:
            ax.axis("off")
            continue
        lo, hi = np.percentile(pooled, [0.5, 99.5])
        if not np.isfinite([lo, hi]).all() or hi <= lo:
            lo, hi = float(pooled.min()), float(pooled.max()) or 1.0
        bins = np.linspace(lo, hi, 60)
        for ds in datasets:
            vals = _values(stats_df, ds, feat)
            if len(vals):
                ax.hist(vals, bins=bins, density=True, histtype="step",
                        linewidth=1.4, label=ds, color=palette[ds])
        ax.set_title(feat, fontsize=10)
        ax.set_ylabel("density", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.15)
        _finish(ax)

    for ax in list(axes)[len(features):]:
        ax.axis("off")
    handles, labels = list(axes)[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    fig.suptitle("Descriptor densities by dataset (shared bins, 0.5-99.5 percentile)", fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    return save_pdf(fig, out_path)


def plot_descriptor_violins(stats_df: pd.DataFrame, features: Sequence[str], out_path: Path,
                             palette: Dict[str, tuple], max_points: int = 20000,
                             seed: int = 1234) -> Path:
    datasets = sorted(stats_df["dataset"].unique())
    rng = np.random.default_rng(seed)
    fig, axes, _ = _grid(len(features))

    for ax, feat in zip(axes, features):
        data, labels = [], []
        for ds in datasets:
            vals = _values(stats_df, ds, feat)
            if len(vals) < 2:
                continue
            if len(vals) > max_points:
                vals = rng.choice(vals, max_points, replace=False)
            data.append(vals)
            labels.append(ds)
        if not data:
            ax.axis("off")
            continue
        parts = ax.violinplot(data, showmedians=True, showextrema=False)
        for body, ds in zip(parts["bodies"], labels):
            body.set_facecolor(palette[ds])
            body.set_alpha(0.7)
        if "cmedians" in parts:
            parts["cmedians"].set_color("black")
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.set_title(feat, fontsize=10)
        ax.grid(True, axis="y", alpha=0.2)
        _finish(ax)

    for ax in list(axes)[len(features):]:
        ax.axis("off")
    fig.suptitle("Descriptor distribution shapes by dataset", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return save_pdf(fig, out_path)


def plot_descriptor_ecdfs(stats_df: pd.DataFrame, features: Sequence[str], out_path: Path,
                           palette: Dict[str, tuple], max_points: int = 50000,
                           seed: int = 1234) -> Path:
    """ECDFs make the dataset-to-dataset gap readable directly: the vertical
    distance between two curves at its widest IS the KS statistic reported in
    the divergence table."""
    datasets = sorted(stats_df["dataset"].unique())
    rng = np.random.default_rng(seed)
    fig, axes, _ = _grid(len(features))

    for ax, feat in zip(axes, features):
        drawn = False
        for ds in datasets:
            vals = _values(stats_df, ds, feat)
            if len(vals) < 2:
                continue
            if len(vals) > max_points:
                vals = rng.choice(vals, max_points, replace=False)
            v = np.sort(vals)
            ax.plot(v, np.arange(1, len(v) + 1) / len(v), label=ds, color=palette[ds], linewidth=1.4)
            drawn = True
        if not drawn:
            ax.axis("off")
            continue
        pooled = stats_df[feat].dropna().to_numpy()
        if len(pooled):
            ax.set_xlim(*np.percentile(pooled, [0.5, 99.5]))
        ax.set_title(feat, fontsize=10)
        ax.set_ylabel("ECDF", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.15)
        _finish(ax)

    for ax in list(axes)[len(features):]:
        ax.axis("off")
    handles, labels = list(axes)[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    fig.suptitle("Descriptor ECDFs by dataset (max gap = KS statistic)", fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    return save_pdf(fig, out_path)


# -----------------------------------------------------------------------------
# Divergence
# -----------------------------------------------------------------------------

def plot_divergence_heatmap(divergence_df: pd.DataFrame, out_path: Path,
                             value: str = "standardized_diff", top_n: int = 30) -> Path:
    """Feature x dataset-pair heatmap, features ordered by how strongly they
    separate any pair, so the top of the chart is the answer to "what actually
    distinguishes these sources"."""
    if divergence_df.empty:
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.axis("off")
        ax.text(0.5, 0.5, "no divergence data", ha="center", va="center")
        return save_pdf(fig, out_path)
    d = divergence_df.copy()
    d["pair"] = d["dataset_a"] + " vs " + d["dataset_b"]
    order = (d.groupby("feature")[value].apply(lambda s: s.abs().max())
             .sort_values(ascending=False).head(top_n).index)
    mat = d[d["feature"].isin(order)].pivot(index="feature", columns="pair", values=value).loc[order]

    diverging = value == "standardized_diff"
    vmax = float(np.nanmax(np.abs(mat.to_numpy()))) or 1.0
    fig, ax = plt.subplots(figsize=(1.9 * mat.shape[1] + 4.5, 0.32 * mat.shape[0] + 2.0))
    im = ax.imshow(mat.to_numpy(), cmap="RdBu_r" if diverging else "viridis",
                   vmin=-vmax if diverging else 0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(mat.columns, rotation=20, ha="right", fontsize=9)
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels(mat.index, fontsize=8)
    label = "standardized mean difference (a - b)" if diverging else "KS statistic"
    ax.set_title(f"Dataset divergence: top {len(order)} features by |{value}|", fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.8, label=label)
    fig.tight_layout()
    return save_pdf(fig, out_path)


def plot_descriptor_correlations(stats_df: pd.DataFrame, features: Sequence[str],
                                  out_path: Path) -> Path:
    """One correlation matrix per dataset, so structural coupling between
    properties can be compared across sources (not just their marginals)."""
    datasets = sorted(stats_df["dataset"].unique())
    feats = [f for f in features if f in stats_df.columns]
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.6 * len(datasets), 4.9), squeeze=False)

    for ax, ds in zip(axes[0], datasets):
        sub = stats_df.loc[stats_df["dataset"] == ds, feats]
        corr = sub.corr().to_numpy()
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(feats)))
        ax.set_xticklabels(feats, rotation=90, fontsize=6)
        ax.set_yticks(range(len(feats)))
        ax.set_yticklabels(feats, fontsize=6)
        ax.set_title(ds, fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.75)
    fig.suptitle("Within-dataset descriptor correlations (Pearson r)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return save_pdf(fig, out_path)


# -----------------------------------------------------------------------------
# Composition / rings / stereochemistry
# -----------------------------------------------------------------------------

def plot_element_composition(stats_df: pd.DataFrame, out_path: Path,
                              palette: Dict[str, tuple]) -> Path:
    """Two views: how OFTEN each heteroatom appears, and how MANY there are
    when it does. A source can hit similar presence rates with very different
    counts per molecule."""
    datasets = sorted(stats_df["dataset"].unique())
    elements = [c for c in stats_df.columns if c.startswith("n_") and len(c.split("_")[1]) <= 2
                and c.split("_")[1] in ("N", "O", "S", "P", "F", "Cl", "Br", "I")]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    width = 0.8 / max(len(datasets), 1)
    x = np.arange(len(elements))

    for i, ds in enumerate(datasets):
        sub = stats_df[stats_df["dataset"] == ds]
        presence = [float((sub[e] > 0).mean()) for e in elements]
        mean_when = [float(sub.loc[sub[e] > 0, e].mean()) if (sub[e] > 0).any() else 0.0
                     for e in elements]
        axes[0].bar(x + i * width, presence, width, label=ds, color=palette[ds])
        axes[1].bar(x + i * width, mean_when, width, label=ds, color=palette[ds])

    for ax, title, ylab in ((axes[0], "Fraction of molecules containing element", "fraction"),
                             (axes[1], "Mean count when present", "atoms / molecule")):
        ax.set_xticks(x + width * (len(datasets) - 1) / 2)
        ax.set_xticklabels([e.split("_", 1)[1] for e in elements])
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylab)
        ax.grid(True, axis="y", alpha=0.2)
        _finish(ax)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return save_pdf(fig, out_path)


def plot_ring_profile(stats_df: pd.DataFrame, out_path: Path, palette: Dict[str, tuple]) -> Path:
    datasets = sorted(stats_df["dataset"].unique())
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5))

    for ax, feat, title in ((axes[0][0], "n_aromatic_rings", "Aromatic rings"),
                             (axes[0][1], "n_aliphatic_rings", "Aliphatic rings"),
                             (axes[0][2], "max_ring_size", "Largest ring size")):
        for ds in datasets:
            vals = _values(stats_df, ds, feat)
            if not len(vals):
                continue
            hi = int(np.percentile(vals, 99.5))
            bins = np.arange(-0.5, max(hi, 1) + 1.5)
            ax.hist(vals, bins=bins, density=True, histtype="step", linewidth=1.4,
                    label=ds, color=palette[ds])
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("density", fontsize=8)
        ax.grid(True, alpha=0.15)
        _finish(ax)

    motifs = [("n_macrocycles", "has macrocycle (>=12)"), ("n_spiro_atoms", "has spiro atom"),
              ("n_bridgehead_atoms", "has bridgehead"), ("n_rings", "acyclic (0 rings)")]
    x = np.arange(len(motifs))
    width = 0.8 / max(len(datasets), 1)
    for i, ds in enumerate(datasets):
        sub = stats_df[stats_df["dataset"] == ds]
        fr = []
        for feat, _ in motifs:
            fr.append(float((sub[feat] == 0).mean()) if feat == "n_rings"
                      else float((sub[feat] > 0).mean()))
        axes[1][0].bar(x + i * width, fr, width, label=ds, color=palette[ds])
    axes[1][0].set_xticks(x + width * (len(datasets) - 1) / 2)
    axes[1][0].set_xticklabels([m[1] for m in motifs], rotation=20, ha="right", fontsize=8)
    axes[1][0].set_title("Ring-motif prevalence", fontsize=10)
    axes[1][0].set_ylabel("fraction of molecules")
    axes[1][0].grid(True, axis="y", alpha=0.2)
    _finish(axes[1][0])

    for ax, feat, title in ((axes[1][1], "FractionCSP3", "FractionCSP3 (sp3 character)"),
                             (axes[1][2], "frac_carbon", "Carbon fraction of heavy atoms")):
        for ds in datasets:
            vals = _values(stats_df, ds, feat)
            if len(vals):
                ax.hist(vals, bins=np.linspace(0, 1, 50), density=True, histtype="step",
                        linewidth=1.4, label=ds, color=palette[ds])
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("density", fontsize=8)
        ax.grid(True, alpha=0.15)
        _finish(ax)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    fig.suptitle("Ring topology and saturation by dataset", fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    return save_pdf(fig, out_path)


def plot_stereochemistry(stats_df: pd.DataFrame, out_path: Path, palette: Dict[str, tuple]) -> Path:
    """Stereocenter counts and how many are actually SPECIFIED -- the sharpest
    curated-natural-product vs reaction-corpus signal there is."""
    datasets = sorted(stats_df["dataset"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))

    for ds in datasets:
        vals = _values(stats_df, ds, "n_stereocenters")
        if len(vals):
            hi = int(np.percentile(vals, 99))
            axes[0].hist(vals, bins=np.arange(-0.5, max(hi, 1) + 1.5), density=True,
                         histtype="step", linewidth=1.4, label=ds, color=palette[ds])
    axes[0].set_title("Stereocenters per molecule", fontsize=10)
    axes[0].set_ylabel("density")

    x = np.arange(2)
    width = 0.8 / max(len(datasets), 1)
    for i, ds in enumerate(datasets):
        sub = stats_df[stats_df["dataset"] == ds]
        has_any = float((sub["n_stereocenters"] > 0).mean())
        fully = float((sub.loc[sub["n_stereocenters"] > 0, "frac_stereo_specified"] == 1).mean()) \
            if (sub["n_stereocenters"] > 0).any() else 0.0
        axes[1].bar(x + i * width, [has_any, fully], width, label=ds, color=palette[ds])
    axes[1].set_xticks(x + width * (len(datasets) - 1) / 2)
    axes[1].set_xticklabels(["has >=1 stereocenter", "fully specified\n(of those)"], fontsize=8)
    axes[1].set_title("Stereochemistry presence and completeness", fontsize=10)
    axes[1].set_ylabel("fraction")

    for ds in datasets:
        vals = _values(stats_df, ds, "frac_stereo_specified")
        if len(vals):
            axes[2].hist(vals, bins=np.linspace(0, 1, 40), density=True, histtype="step",
                          linewidth=1.4, label=ds, color=palette[ds])
    axes[2].set_title("Fraction of stereocenters specified", fontsize=10)
    axes[2].set_ylabel("density")

    for ax in axes:
        ax.grid(True, alpha=0.18)
        _finish(ax)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Stereochemistry by dataset", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save_pdf(fig, out_path)


# -----------------------------------------------------------------------------
# Scaffolds / overlap / chemical-space shape
# -----------------------------------------------------------------------------

def plot_scaffold_diversity(scaffold_summary_df: pd.DataFrame, coverage_df: pd.DataFrame,
                             out_path: Path, palette: Dict[str, tuple]) -> Path:
    datasets = sorted(coverage_df["dataset"].unique()) if not coverage_df.empty else []
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))

    for ds in datasets:
        curve = coverage_df[coverage_df["dataset"] == ds].sort_values("frac_scaffolds")
        axes[0].plot(curve["frac_scaffolds"], curve["frac_molecules"], label=ds,
                     color=palette[ds], linewidth=1.6)
    axes[0].plot([0, 1], [0, 1], color="0.6", linestyle="--", linewidth=1, label="uniform")
    axes[0].set_xlabel("fraction of scaffolds (most common first)")
    axes[0].set_ylabel("fraction of molecules covered")
    axes[0].set_title("Scaffold concentration", fontsize=10)
    axes[0].legend(frameon=False, fontsize=8)

    if not scaffold_summary_df.empty:
        x = np.arange(len(scaffold_summary_df))
        axes[1].bar(x, scaffold_summary_df["scaffolds_per_molecule"],
                    color=[palette.get(d, "0.5") for d in scaffold_summary_df["dataset"]])
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(scaffold_summary_df["dataset"], rotation=20, ha="right")
        axes[1].set_ylabel("unique scaffolds / molecule")
        axes[1].set_title("Scaffold diversity (higher = less repetitive)", fontsize=10)

        width = 0.35
        axes[2].bar(x - width / 2, scaffold_summary_df["top1pct_scaffold_coverage"], width,
                    label="top 1% of scaffolds", color="#4C78A8")
        axes[2].bar(x + width / 2, scaffold_summary_df["top10pct_scaffold_coverage"], width,
                    label="top 10% of scaffolds", color="#9ECAE1")
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(scaffold_summary_df["dataset"], rotation=20, ha="right")
        axes[2].set_ylabel("fraction of molecules covered")
        axes[2].set_title("Coverage by the most common scaffolds", fontsize=10)
        axes[2].legend(frameon=False, fontsize=8)

    for ax in axes:
        ax.grid(True, alpha=0.18)
        _finish(ax)
    fig.suptitle("Murcko scaffold diversity by dataset", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save_pdf(fig, out_path)


def plot_dataset_overlap(overlap_pairs_df: pd.DataFrame, overlap_summary_df: pd.DataFrame,
                          out_path: Path) -> Path:
    """Exact cross-source molecule sharing (canonical SMILES, full corpus)."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))

    if not overlap_pairs_df.empty:
        labels = overlap_pairs_df["dataset_a"] + "\nvs " + overlap_pairs_df["dataset_b"]
        x = np.arange(len(overlap_pairs_df))
        axes[0].bar(x, overlap_pairs_df["n_shared"], color="#4C78A8")
        for xi, n in zip(x, overlap_pairs_df["n_shared"]):
            axes[0].text(xi, n, f"{int(n):,}", ha="center", va="bottom", fontsize=8)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels, fontsize=8)
        axes[0].set_ylabel("shared molecules")
        axes[0].set_title("Molecules shared between sources", fontsize=10)
        axes[0].margins(y=0.15)

        axes[1].bar(x, overlap_pairs_df["jaccard"], color="#F58518")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, fontsize=8)
        axes[1].set_ylabel("Jaccard index")
        axes[1].set_title("Overlap, size-normalized", fontsize=10)

    if not overlap_summary_df.empty:
        x = np.arange(len(overlap_summary_df))
        width = 0.35
        axes[2].bar(x - width / 2, overlap_summary_df["internal_duplicate_frac"], width,
                    label="internal duplicates", color="#E45756")
        axes[2].bar(x + width / 2, overlap_summary_df["frac_shared_with_others"], width,
                    label="shared with another source", color="#72B7B2")
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(overlap_summary_df["dataset"], rotation=20, ha="right")
        axes[2].set_ylabel("fraction")
        axes[2].set_title("Redundancy within / across sources", fontsize=10)
        axes[2].legend(frameon=False, fontsize=8)

    for ax in axes:
        ax.grid(True, axis="y", alpha=0.2)
        _finish(ax)
    fig.suptitle("Cross-dataset molecule overlap (exact, canonical SMILES, full corpus)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save_pdf(fig, out_path)


def plot_chemical_space_hexbin(stats_df: pd.DataFrame, out_path: Path,
                                x_feat: str = "MolWt", y_feat: str = "MolLogP") -> Path:
    """Where each source sits in the classic size-vs-lipophilicity plane --
    the joint structure the per-feature marginals above cannot show."""
    datasets = sorted(stats_df["dataset"].unique())
    pooled_x = stats_df[x_feat].dropna()
    pooled_y = stats_df[y_feat].dropna()
    xlim = tuple(np.percentile(pooled_x, [0.5, 99.5]))
    ylim = tuple(np.percentile(pooled_y, [0.5, 99.5]))

    fig, axes = plt.subplots(1, len(datasets), figsize=(4.6 * len(datasets), 4.4),
                              squeeze=False, sharex=True, sharey=True)
    for ax, ds in zip(axes[0], datasets):
        sub = stats_df[stats_df["dataset"] == ds][[x_feat, y_feat]].dropna()
        hb = ax.hexbin(sub[x_feat], sub[y_feat], gridsize=60, bins="log",
                       extent=(xlim[0], xlim[1], ylim[0], ylim[1]), cmap="Blues", mincnt=1)
        ax.set_title(f"{ds}  (n={len(sub):,})", fontsize=10)
        ax.set_xlabel(x_feat)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        _finish(ax)
        fig.colorbar(hb, ax=ax, shrink=0.8, label="log10(count)")
    axes[0][0].set_ylabel(y_feat)
    fig.suptitle(f"Chemical-space occupancy: {y_feat} vs {x_feat}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save_pdf(fig, out_path)
