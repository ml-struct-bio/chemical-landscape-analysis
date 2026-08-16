#!/usr/bin/env python
"""
07_layer_comparison.py
=======================

Figures for the decoder layer-comparison analysis. Reads
`data/07_layer_comparison/<tag>/` and writes PDFs to
`figures/07_layer_comparison/<tag>/`.

Runs on its own -- no --data-dir, no re-loading the layerwise/global_cond
files, no recomputing metrics. Which representations were compared, which
layers got traversal filmstrips, and the traversal step resolution were all
decided by `analysis/07_layer_comparison.py`; iterating on layout here is
free.

Figures
-------
    property_r2_vs_layer.pdf            property linear-decodability sweep
    cluster_quality_vs_layer.pdf        Butina cluster-quality sweep (only if
                                         the analysis was given --cluster-tag)
    nn_overlap_vs_layer.pdf             ECFP-vs-embedding NN-agreement sweep
    pc12_interpretability_vs_layer.pdf  PC1/PC2-vs-descriptor |r| sweep
    <representation>_pc<k>_traversal.pdf   PC-traversal filmstrips, one per
                                            selected representation x PC

Usage
-----
    python plotting/07_layer_comparison.py --tag main
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.common.manifest import require_manifest  # noqa: E402
from src.common.paths import figures_dir  # noqa: E402
from src.plotting.layer_comparison import plot_layer_comparison  # noqa: E402
from src.plotting.style import apply_style  # noqa: E402

SLUG = "07_layer_comparison"
SCHEMA_VERSION = 1


def parse_args():
    p = argparse.ArgumentParser(
        description="Draw the decoder layer-comparison figures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--tag", type=str, default="main",
                   help="Which analysis output to draw, i.e. data/07_layer_comparison/<tag>/.")
    return p.parse_args()


def main():
    args = parse_args()
    apply_style()

    directory, manifest = require_manifest(SLUG, args.tag, schema_version=SCHEMA_VERSION)
    params = manifest.get("params", {})
    out_dir = figures_dir(SLUG, args.tag, create=True)

    print(f"### {args.tag} -- {params.get('n_molecules', '?')} molecules, "
          f"{len(params.get('representations', []))} representations "
          f"(stream={params.get('stream', '?')}) ###")

    written = plot_layer_comparison(directory, out_dir, params=params)
    for key, path in written.items():
        print(f"Saved {path}")
    print(f"\nWrote {len(written)} figures to {out_dir}")


if __name__ == "__main__":
    main()
