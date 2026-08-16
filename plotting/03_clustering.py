#!/usr/bin/env python
"""
03_clustering.py
================

Figures for the Butina clustering. Reads `data/03_clustering/<tag>/` and writes
PDFs to `figures/03_clustering/<tag>/`.

Runs on its own -- no `--data-dir`, no re-clustering. Which clusters appear,
which molecules represent them, and the bar normalization were all decided by
`analysis/03_clustering.py`, so iterating on layout is just re-running this.

Figures
-------
    cluster_representatives.pdf   structures + mean-descriptor bars per cluster
    cluster_sizes.pdf             cluster-size distribution + mean-property
                                   distributions, over ALL clusters

Usage
-----
    python plotting/03_clustering.py --tag main
    python plotting/03_clustering.py --tag main --mols-per-row 2 --sub-img-size 220 220
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd  # noqa: E402

from src.common.manifest import require_manifest  # noqa: E402
from src.common.paths import figures_dir  # noqa: E402
from src.plotting.clustering import (  # noqa: E402
    cluster_representatives_figure,
    cluster_size_figure,
)
from src.plotting.style import apply_style  # noqa: E402

SLUG = "03_clustering"
SCHEMA_VERSION = 1
ALL_FIGURES = ("representatives", "sizes")


def parse_args():
    p = argparse.ArgumentParser(
        description="Draw the Butina clustering figures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--tag", type=str, default="main",
                   help="Which analysis output to draw, i.e. data/03_clustering/<tag>/.")
    p.add_argument("--figures", nargs="+", default=["all"], choices=["all", *ALL_FIGURES])
    p.add_argument("--mols-per-row", type=int, default=3,
                   help="Molecules per row inside each cluster's structure panel.")
    p.add_argument("--sub-img-size", type=int, nargs=2, default=[180, 180],
                   help="Per-molecule panel size in pixels.")
    p.add_argument("--ncols", type=int, default=5,
                   help="Cluster cells per row in the representatives figure.")
    p.add_argument("--max-clusters", type=int, default=None,
                   help="Draw only the largest N of the clusters the analysis prepared. "
                        "Default: all of them.")
    p.add_argument("--size-bins", type=int, default=30,
                   help="Histogram bins in the cluster-size/mean-property distribution figure.")
    return p.parse_args()


def main():
    args = parse_args()
    apply_style()

    directory, manifest = require_manifest(SLUG, args.tag, schema_version=SCHEMA_VERSION)
    params = manifest.get("params", {})
    out_dir = figures_dir(SLUG, args.tag, create=True)
    wanted = set(ALL_FIGURES) if "all" in args.figures else set(args.figures)
    props = params.get("props", ["MolWt", "LogP", "TPSA", "Rings"])

    print(f"### {args.tag} -- {params.get('n_clusters', '?')} clusters over "
          f"{params.get('n_molecules', '?')} molecules "
          f"(cutoff {params.get('butina_cutoff', '?')}) ###")

    if "representatives" in wanted:
        plot_df = pd.read_csv(directory / "plot_clusters.csv")
        if args.max_clusters is not None:
            plot_df = plot_df.head(args.max_clusters)
        reps_df = pd.read_csv(directory / "representatives.csv").sort_values(["cluster", "rank"])
        reps = {int(c): g["smiles"].tolist() for c, g in reps_df.groupby("cluster")}
        path = cluster_representatives_figure(
            plot_df, reps, props, out_dir / "cluster_representatives.pdf",
            mols_per_row=args.mols_per_row, sub_img_size=tuple(args.sub_img_size),
            ncols=args.ncols)
        print(f"Saved {path}")

    if "sizes" in wanted:
        stats_df = pd.read_csv(directory / "cluster_stats.csv")
        path = cluster_size_figure(stats_df, out_dir / "cluster_sizes.pdf",
                                   props=props, bins=args.size_bins)
        print(f"Saved {path}")

    print(f"Done -> {out_dir}")


if __name__ == "__main__":
    main()
