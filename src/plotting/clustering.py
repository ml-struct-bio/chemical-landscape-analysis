"""The Butina cluster-summary figure.

Consumes `data/03_clustering/<tag>/` only. Which clusters appear, which
molecules represent them, and the bar normalization were all decided by
`src/analysis/clustering.py`.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.plotting.mol_render import render_mol_grid
from src.plotting.style import save_pdf


def cluster_representatives_figure(plot_df: pd.DataFrame, reps: Dict[int, List[str]],
                                   props: Sequence[str], out_path: Path,
                                   *, mols_per_row: int = 3,
                                   sub_img_size: tuple = (180, 180),
                                   ncols: int = 5) -> Path:
    """One cell per cluster: a grid of representative structures above a bar
    row of that cluster's mean descriptor values.

    Bars use the pre-computed `<prop>_norm` columns, min-maxed across the
    plotted clusters, so a full bar means "highest among these clusters" rather
    than anything absolute.
    """
    n = len(plot_df)
    nrows = math.ceil(n / ncols)
    fig = plt.figure(figsize=(14, 4 * nrows))
    outer = fig.add_gridspec(nrows, ncols, wspace=0.1, hspace=0.4)
    norm_cols = [f"{p}_norm" for p in props]

    for k, (_, row) in enumerate(plot_df.iterrows()):
        r, c = divmod(k, ncols)
        gs = outer[r, c].subgridspec(2, 1, height_ratios=[2.5, 1], hspace=0.02)
        ax_img = fig.add_subplot(gs[0])
        ax_bar = fig.add_subplot(gs[1])

        cluster = int(row["cluster"])
        rep_smiles = reps.get(cluster, [])
        if rep_smiles:
            ax_img.imshow(np.asarray(render_mol_grid(rep_smiles, mols_per_row=mols_per_row,
                                                     sub_img_size=sub_img_size)))
        else:
            ax_img.text(0.5, 0.5, "no representatives", ha="center", va="center",
                        fontsize=8, color="0.5", transform=ax_img.transAxes)
        ax_img.axis("off")
        ax_img.set_title(f"Cluster {cluster} (n={int(row['n_molecules'])})", fontsize=10)

        ax_bar.bar(range(len(props)), [row[c] for c in norm_cols])
        ax_bar.set_ylim(0, 1)
        ax_bar.set_xticks(range(len(props)))
        ax_bar.set_xticklabels(props, rotation=45, fontsize=8)
        ax_bar.set_yticks([])

    for k in range(n, nrows * ncols):
        r, c = divmod(k, ncols)
        gs = outer[r, c].subgridspec(2, 1)
        fig.add_subplot(gs[0]).axis("off")
        fig.add_subplot(gs[1]).axis("off")

    return save_pdf(fig, out_path)


def cluster_size_figure(stats_df: pd.DataFrame, out_path: Path,
                        props: Sequence[str] = ("MolWt", "LogP", "TPSA", "Rings"),
                        bins: int = 30) -> Path:
    """Cluster-level distributions: how big clusters are, and how each
    cluster's mean descriptor value is spread across clusters.

    Not in the previous pipeline. Butina's output is dominated by a long tail of
    singletons, and the representatives figure only ever shows the largest ~24
    clusters -- so nothing in the old output revealed the shape of the sizes or
    the property averages across the FULL set of clusters, only the largest few.
    """
    props = [p for p in props if p in stats_df.columns]
    sizes = stats_df["n_molecules"].to_numpy()
    total = int(sizes.sum())
    n_singleton = int((sizes == 1).sum())

    ncols = 1 + len(props)
    fig, axes = plt.subplots(1, ncols, figsize=(2.9 * ncols, 3.6), constrained_layout=True)

    size_bins = np.logspace(0, np.log10(max(int(sizes.max()), 1)), bins)
    axes[0].hist(sizes, bins=size_bins, color="#4363d8")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("cluster size (molecules)")
    axes[0].set_ylabel("n clusters")
    axes[0].set_title(f"{len(stats_df)} clusters, {n_singleton:,} singletons\n"
                      f"over {total:,} molecules", fontsize=10)

    for ax, prop in zip(axes[1:], props):
        vals = stats_df[prop].dropna().to_numpy()
        ax.hist(vals, bins=bins, color="#4363d8")
        ax.set_xlabel(f"mean {prop} per cluster")
        ax.set_ylabel("n clusters")
        ax.set_title(f"cluster-average {prop}", fontsize=10)

    return save_pdf(fig, out_path)
