#!/usr/bin/env python
"""
04_joint_pca.py
===============

Figures for the joint structural + spectral PCA. Reads
`data/04_joint_pca/<tag>/` and writes PDFs to `figures/04_joint_pca/<tag>/`.

Runs on its own -- no `--data-dir`, no checkpoint, no re-fitting. Every molecule
choice, percentile and correlation was decided by `analysis/04_joint_pca.py`.
Iterating on cosmetics is just re-running this.

Figures
-------
    pc{i}_joint_traversal.pdf       1H sticks / 13C sticks / structure / scatter
    descriptor_correlations.pdf     each PC vs its best RDKit descriptor
    spectral_correlations.pdf       each PC vs its best NMR spectral feature
    spectral_correlation_heatmap.pdf   full PC x spectral-feature matrix
    descriptor_correlation_heatmap.pdf full PC x descriptor matrix

Usage
-----
    python plotting/04_joint_pca.py --tag global_cond
    python plotting/04_joint_pca.py --tag decoder_x_L05_t0.001 --figures traversal
    python plotting/04_joint_pca.py --all-tags
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.common.constants import DEFAULT_C_PPM_RANGE, DEFAULT_H_PPM_RANGE  # noqa: E402
from src.common.manifest import require_manifest  # noqa: E402
from src.common.paths import DATA_ROOT, figures_dir  # noqa: E402
from src.plotting.joint_pca import (  # noqa: E402
    StepPeaks,
    correlation_grid,
    correlation_heatmap,
    joint_traversal_figure,
)
from src.plotting.style import apply_style  # noqa: E402

SLUG = "04_joint_pca"
SCHEMA_VERSION = 1
ALL_FIGURES = ("traversal", "descriptor-correlations", "spectral-correlations", "heatmaps")


def parse_args():
    p = argparse.ArgumentParser(
        description="Draw the joint structural + spectral PCA figures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--tag", type=str, default="global_cond",
                   help="Which analysis output to draw, i.e. data/04_joint_pca/<tag>/.")
    p.add_argument("--all-tags", action="store_true",
                   help="Draw every tag present under data/04_joint_pca/.")
    p.add_argument("--figures", nargs="+", default=["all"],
                   choices=["all", *ALL_FIGURES],
                   help="Subset of figures to draw.")
    p.add_argument("--pcs", nargs="+", type=int, default=None,
                   help="Only draw traversals for these PCs (1-based). Default: all.")
    p.add_argument("--mol-size", type=int, default=260,
                   help="Rendered structure size in pixels.")
    p.add_argument("--highlight-color", type=str, default="red",
                   help="Colour of the dot marking the step molecule. Note the spectra rows "
                        "already use blue for 1H and red for 13C, so 'red'/'blue' will echo "
                        "one of them.")
    p.add_argument("--corr-alpha", type=float, default=0.02,
                   help="Point opacity in the correlation scatters. Every stored molecule is "
                        "drawn, so this has to be low to show density rather than a blob.")
    return p.parse_args()


def draw_tag(tag: str, args) -> None:
    directory, manifest = require_manifest(SLUG, tag, schema_version=SCHEMA_VERSION)
    params = manifest.get("params", {})
    out_dir = figures_dir(SLUG, tag, create=True)
    wanted = set(ALL_FIGURES) if "all" in args.figures else set(args.figures)

    print(f"\n### {tag} -- {params.get('embedding', '?')} "
          f"({params.get('n_molecules', '?')} molecules) ###")

    pct_lo = float(params.get("pct_lo", 1.0))
    pct_hi = float(params.get("pct_hi", 99.0))
    h_range = tuple(params.get("h_ppm_range", DEFAULT_H_PPM_RANGE))
    c_range = tuple(params.get("c_ppm_range", DEFAULT_C_PPM_RANGE))

    if "traversal" in wanted:
        peaks = StepPeaks(directory / "traversal_peaks.npz")
        if not peaks.ok:
            print("[note] This analysis stored no peak lists, so the traversal filmstrips "
                  "(which draw a real molecule's spectrum) are skipped. Re-run "
                  "extraction/01_spectral_features.py, then the analysis, to add them.")
        else:
            traversal = pd.read_csv(directory / "traversal.csv")
            stats = pd.read_csv(directory / "pc_stats.csv").set_index("pc")
            backdrop = np.load(directory / "pcs_background.npz")["coords"]
            bars_path = directory / "traversal_bars.csv"
            bars_all = pd.read_csv(bars_path) if bars_path.exists() else None

            pcs = args.pcs or sorted(traversal["pc"].unique())
            for pc in pcs:
                steps = traversal[traversal["pc"] == pc].sort_values("step")
                if steps.empty:
                    print(f"[warn] no traversal rows for PC{pc} -- skipping.")
                    continue
                bars = bars_all[bars_all["pc"] == pc] if bars_all is not None else None
                path = joint_traversal_figure(
                    pc, steps, stats.loc[pc], peaks, backdrop, bars,
                    out_dir / f"pc{pc}_joint_traversal.pdf",
                    h_ppm_range=h_range, c_ppm_range=c_range, mol_size=args.mol_size,
                    highlight_color=args.highlight_color, pct_lo=pct_lo, pct_hi=pct_hi)
                print(f"Saved {path}")

    scatter = None
    if wanted & {"descriptor-correlations", "spectral-correlations"}:
        scatter = np.load(directory / "corr_scatter.npz")
        drawn, total = len(scatter["pc_values"]), int(scatter["n_total"])
        if drawn < total:
            print(f"[note] correlation scatters draw {drawn}/{total} molecules "
                  f"(--corr-scatter-max at analysis time); reported r values used all "
                  f"{total}.")

    if "descriptor-correlations" in wanted:
        best = pd.read_csv(directory / "pc_best_descriptor.csv")
        path = correlation_grid(scatter["pc_values"], scatter["descriptor_values"], best,
                                 "descriptor", out_dir / "descriptor_correlations.pdf",
                                 alpha=args.corr_alpha)
        print(f"Saved {path}")

    if "spectral-correlations" in wanted:
        best = pd.read_csv(directory / "pc_best_spectral_feature.csv")
        path = correlation_grid(scatter["pc_values"], scatter["spectral_values"], best,
                                 "feature", out_dir / "spectral_correlations.pdf",
                                 alpha=args.corr_alpha)
        print(f"Saved {path}")

    if "heatmaps" in wanted:
        for name, title in (("spectral", "Pearson r: embedding PCs vs. NMR spectral features"),
                            ("descriptor", "Pearson r: embedding PCs vs. RDKit descriptors")):
            corr = pd.read_csv(directory / f"{name}_correlations.csv", index_col=0)
            path = correlation_heatmap(corr, out_dir / f"{name}_correlation_heatmap.pdf", title)
            print(f"Saved {path}")


def main():
    args = parse_args()
    apply_style()

    if args.all_tags:
        root = DATA_ROOT / SLUG
        tags = sorted(p.name for p in root.iterdir() if p.is_dir()) if root.exists() else []
        if not tags:
            raise SystemExit(f"No analysis output under {root}.\n"
                             f"Run: python analysis/{SLUG}.py --data-dir ... --embeddings all")
    else:
        tags = [args.tag]

    for tag in tags:
        draw_tag(tag, args)
    print(f"\nDone. {len(tags)} tag(s) -> figures/{SLUG}/")


if __name__ == "__main__":
    main()
