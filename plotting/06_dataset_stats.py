#!/usr/bin/env python
"""
06_dataset_stats.py
====================

Figures for the per-dataset molecular characterization. Reads
`data/06_dataset_stats/<tag>/` and writes PDFs to `figures/06_dataset_stats/<tag>/`.

Runs on its own -- no `--data-dir`, no recomputation. Every dataset color
resolves through `configs/colors.yaml` plus `--dataset-colors`, exactly like
scripts 05's `--color-by dataset`, so a source keeps the same color in every
figure of the pipeline.

Figures
-------
    dataset_counts.pdf            dataset sizes by split
    descriptor_boxplots.pdf        headline descriptors, one box per dataset
    descriptor_histograms.pdf      headline descriptors, shared bins/density
    descriptor_violins.pdf         headline descriptors, distribution shape
    descriptor_ecdfs.pdf           headline descriptors, ECDFs (max gap = KS)
    descriptor_correlations.pdf    within-dataset descriptor correlation matrices
    divergence_smd.pdf             feature x pair heatmap, standardized mean diff
    divergence_ks.pdf              feature x pair heatmap, KS statistic
    element_composition.pdf        heteroatom presence/count by dataset
    ring_profile.pdf               ring topology and saturation by dataset
    stereochemistry.pdf            stereocenter counts and specification rate
    scaffold_diversity.pdf         Murcko scaffold concentration/diversity
    dataset_overlap.pdf            exact cross-dataset molecule sharing
    chemical_space_hexbin.pdf      MolWt vs MolLogP occupancy per dataset

Usage
-----
    python plotting/06_dataset_stats.py --tag main
    python plotting/06_dataset_stats.py --tag main --dataset-colors nmrexp='#4C8DAE'
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
from src.common.palette import add_palette_args, palette_from_args, resolve  # noqa: E402
from src.common.paths import figures_dir  # noqa: E402
from src.plotting.dataset_stats import (  # noqa: E402
    plot_chemical_space_hexbin,
    plot_dataset_counts,
    plot_dataset_overlap,
    plot_descriptor_boxplots,
    plot_descriptor_correlations,
    plot_descriptor_ecdfs,
    plot_descriptor_histograms,
    plot_descriptor_violins,
    plot_divergence_heatmap,
    plot_element_composition,
    plot_ring_profile,
    plot_scaffold_diversity,
    plot_stereochemistry,
)
from src.plotting.style import apply_style  # noqa: E402

SLUG = "06_dataset_stats"
SCHEMA_VERSION = 1
ALL_FIGURES = (
    "counts", "boxplots", "histograms", "violins", "ecdfs", "correlations",
    "divergence", "elements", "rings", "stereo", "scaffolds", "overlap", "hexbin",
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Draw the dataset-statistics figures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--tag", type=str, default="main",
                   help="Which analysis output to draw, i.e. data/06_dataset_stats/<tag>/.")
    p.add_argument("--figures", nargs="+", default=["all"], choices=["all", *ALL_FIGURES])
    p.add_argument("--seed", type=int, default=1234,
                   help="Subsampling seed for the violin/ECDF panels.")
    add_palette_args(p, modes=("dataset",))
    return p.parse_args()


def main():
    args = parse_args()
    apply_style()

    directory, manifest = require_manifest(SLUG, args.tag, schema_version=SCHEMA_VERSION)
    params = manifest.get("params", {})
    out_dir = figures_dir(SLUG, args.tag, create=True)
    wanted = set(ALL_FIGURES) if "all" in args.figures else set(args.figures)

    stats_df = pd.read_csv(directory / "stats.csv")
    datasets = sorted(stats_df["dataset"].unique())
    palette_cfg = palette_from_args(args, root_dir=_REPO_ROOT)
    palette = resolve(datasets, palette_cfg.dataset, mode="dataset", ordered_keys=datasets)

    headline = [f for f in params.get("headline_descriptors", []) if f in stats_df.columns]

    print(f"### {args.tag} -- {params.get('n_total_molecules', '?')} molecules, "
          f"{len(datasets)} datasets ({', '.join(datasets)}); "
          f"{params.get('n_sampled_molecules', '?')} sampled for molecular features ###")

    def emit(key: str, fn, *fn_args, **fn_kwargs) -> None:
        if key not in wanted:
            return
        try:
            path = fn(*fn_args, **fn_kwargs)
            print(f"Saved {path}")
        except Exception as exc:  # one bad panel shouldn't cost the whole run
            print(f"[warn] Could not render '{key}': {type(exc).__name__}: {exc}")

    emit("counts", plot_dataset_counts, pd.read_csv(directory / "counts.csv"),
         out_dir / "dataset_counts.pdf")
    emit("boxplots", plot_descriptor_boxplots, stats_df, headline,
         out_dir / "descriptor_boxplots.pdf", palette)
    emit("histograms", plot_descriptor_histograms, stats_df, headline,
         out_dir / "descriptor_histograms.pdf", palette)
    emit("violins", plot_descriptor_violins, stats_df, headline,
         out_dir / "descriptor_violins.pdf", palette, seed=args.seed)
    emit("ecdfs", plot_descriptor_ecdfs, stats_df, headline,
         out_dir / "descriptor_ecdfs.pdf", palette, seed=args.seed)
    emit("correlations", plot_descriptor_correlations, stats_df, headline,
         out_dir / "descriptor_correlations.pdf")

    if "divergence" in wanted:
        divergence_df = pd.read_csv(directory / "divergence.csv")
        emit("divergence", plot_divergence_heatmap, divergence_df,
             out_dir / "divergence_smd.pdf", value="standardized_diff")
        emit("divergence", plot_divergence_heatmap, divergence_df,
             out_dir / "divergence_ks.pdf", value="ks_stat")

    emit("elements", plot_element_composition, stats_df, out_dir / "element_composition.pdf",
         palette)
    emit("rings", plot_ring_profile, stats_df, out_dir / "ring_profile.pdf", palette)
    emit("stereo", plot_stereochemistry, stats_df, out_dir / "stereochemistry.pdf", palette)

    if "scaffolds" in wanted:
        scaffold_summary_df = pd.read_csv(directory / "scaffold_summary.csv")
        coverage_df = pd.read_csv(directory / "scaffold_coverage.csv")
        emit("scaffolds", plot_scaffold_diversity, scaffold_summary_df, coverage_df,
             out_dir / "scaffold_diversity.pdf", palette)

    if "overlap" in wanted:
        emit("overlap", plot_dataset_overlap, pd.read_csv(directory / "overlap_pairs.csv"),
             pd.read_csv(directory / "overlap_summary.csv"), out_dir / "dataset_overlap.pdf")

    emit("hexbin", plot_chemical_space_hexbin, stats_df, out_dir / "chemical_space_hexbin.pdf")

    print(f"Done -> {out_dir}")


if __name__ == "__main__":
    main()
