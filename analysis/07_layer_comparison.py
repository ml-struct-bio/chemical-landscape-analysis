#!/usr/bin/env python
"""
07_layer_comparison.py
=======================

Compares decoder per-layer hidden-state representations against each other and
against two baselines (ECFP, the peak embedder's `global_cond`): property
linear-decodability, unsupervised cluster-quality vs. Butina labels, ECFP-vs-
embedding nearest-neighbor agreement, and PC1/PC2-vs-RDKit-descriptor
interpretability -- all swept across decoder depth. Also resolves PC-traversal
filmstrip steps for a handful of representative layers.

Port of the previous pipeline's `10_run_layer_comparison_experiment.py`. Same
metrics, same defaults. Two things did NOT come over: property-direction
traversal (this port only does PC traversal, via the same machinery
`04_joint_pca` uses), and the old per-representation PC1/PC2-vs-descriptor
scatter grid (~38 near-duplicate figures on a full sweep) -- the PC-
interpretability SWEEP chart plus the full `pc_correlations.csv` numbers cover
the same ground.

Loading + aligning every (layer, timestep) of one stream is the expensive part
(tens of GB read + a corpus-wide join) and has nothing to do with which
metrics or traversal layers are requested, so it is cached under
`cache/layer_comparison/`. Re-running with different --traversal-layers,
--n-steps or percentiles reuses the cache; --refit forces a redo.

Outputs (`data/07_layer_comparison/<tag>/`)
--------------------------------------------
    metrics.csv              one row per representation: property R^2, cluster
                             quality (if --cluster-tag given), NN overlap,
                             PC1/PC2 best-descriptor + r
    summary.txt               best representation per metric
    pc_correlations.csv       long-format: representation x PC x descriptor x r
    pc_best_descriptor.csv    representation x PC's single best descriptor
    traversal.csv             PC-traversal filmstrip steps (real molecules only)
    traversal_stats.csv       explained variance ratio per traversal PC
    traversal_background_<representation>.npz   subsampled PC-space backdrop

Usage
-----
    python analysis/07_layer_comparison.py \\
        --layerwise-dir /scratch/.../epoch1399 --data-dir /scratch/.../epoch1399 \\
        --prefix cotrain --splits train --stream x_hidden_mean \\
        --cluster-tag main --traversal-layers 1 6 11
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402

from src.analysis.layer_comparison import run_layer_comparison, validate_traversal_keys  # noqa: E402
from src.analysis.layer_comparison_metrics import PROPERTY_FUNCS  # noqa: E402
from src.common.paths import data_dir  # noqa: E402
from src.common.workers import safe_n_workers  # noqa: E402

SLUG = "07_layer_comparison"


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare decoder layers against each other and against ECFP/global_cond "
                    "baselines: property decodability, cluster-quality, and ECFP-vs-embedding "
                    "NN agreement, swept across depth.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-dir", type=Path, required=True,
                   help="Directory holding extraction's <prefix>_<split>_global_cond.pt "
                        "(ECFP/global_cond baselines).")
    p.add_argument("--layerwise-dir", type=Path, default=None,
                   help="Directory holding <prefix>_<split>_layerwise.pt. Default: --data-dir.")
    p.add_argument("--prefix", type=str, default="cotrain")
    p.add_argument("--splits", nargs="+", default=["train"], choices=["train", "val", "test"])
    p.add_argument("--stream", type=str, default="x_hidden_mean",
                   choices=["x_hidden_mean", "y_hidden_mean"],
                   help="Which pooled decoder stream to compare across layers.")
    p.add_argument("--tag", type=str, default="main",
                   help="Output tag, i.e. data/07_layer_comparison/<tag>/.")

    p.add_argument("--cluster-tag", type=str, default=None,
                   help="Tag of a data/03_clustering/<tag>/cluster_labels.npy run over the SAME "
                        "--splits. Enables cluster-quality metrics; skipped if omitted.")
    p.add_argument("--include-silhouette", action="store_true",
                   help="Also compute silhouette score (slower; off by default).")
    p.add_argument("--knn-purity-k", type=int, default=25)

    p.add_argument("--nn-overlap-sample", type=int, default=3000,
                   help="Stratified subsample size for the ECFP-vs-embedding NN-agreement analysis.")
    p.add_argument("--k-neighbors", nargs="+", type=int, default=[5, 10, 25])

    p.add_argument("--n-prop-workers", type=int, default=safe_n_workers())
    p.add_argument("--descriptor-names", nargs="+", default=None,
                   help="RDKit descriptor panel for the PC1/PC2 interpretability correlations "
                        "(computed for every representation). Default: the pipeline-wide "
                        "34-descriptor panel, same as 04_joint_pca.")
    p.add_argument("--n-desc-workers", type=int, default=safe_n_workers())

    p.add_argument("--traversal-layers", nargs="+", default=None,
                   help="Which (layer, timestep) keys to resolve PC-traversal filmstrips for. "
                        "Either 'LAYER,TIMESTEP' (e.g. '0,0.001' '6,0.001' '-1,0.001') or a bare "
                        "'LAYER' (e.g. '1' '6' '11'), which uses that layer's first available "
                        "timestep. Default: auto-pick first/middle/last available layer at the "
                        "first available timestep.")
    p.add_argument("--n-pcs-traversal", type=int, default=2)
    p.add_argument("--n-steps", type=int, default=8)
    p.add_argument("--pc-percentile-lo", type=float, default=1.0)
    p.add_argument("--pc-percentile-hi", type=float, default=99.0)
    p.add_argument("--n-background-scatter", type=int, default=20_000,
                   help="Points kept for each traversal representation's PC-space backdrop "
                        "scatter (thinned; percentiles/steps above use every molecule).")

    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                   help="Only reaches the NN-overlap similarity matrices "
                        "(--nn-overlap-sample x --nn-overlap-sample); CPU is fine.")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--refit", action="store_true",
                   help="Ignore any cached aligned representations under cache/layer_comparison/ "
                        "and reload + realign from scratch.")
    return p.parse_args()


def main():
    args = parse_args()

    # Before anything expensive: a malformed --traversal-layers entry used to
    # surface only after the full load + metric sweep in the previous pipeline.
    try:
        validate_traversal_keys(args.traversal_layers)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}")

    layerwise_dir = args.layerwise_dir or args.data_dir

    print("=" * 78)
    print("Decoder layer comparison")
    print(f"  data dir          : {args.data_dir}")
    print(f"  layerwise dir     : {layerwise_dir}")
    print(f"  prefix/splits     : {args.prefix} / {args.splits}")
    print(f"  stream            : {args.stream}")
    print(f"  properties        : {list(PROPERTY_FUNCS.keys())}")
    print(f"  tag               : {args.tag}")
    print("=" * 78)

    written = run_layer_comparison(
        data_dir=args.data_dir, layerwise_dir=layerwise_dir, prefix=args.prefix,
        splits=args.splits, out_dir=data_dir(SLUG, args.tag, create=True), tag=args.tag, slug=SLUG,
        stream=args.stream, cluster_tag=args.cluster_tag,
        include_silhouette=args.include_silhouette, knn_purity_k=args.knn_purity_k,
        nn_overlap_sample=args.nn_overlap_sample, k_neighbors=args.k_neighbors,
        n_prop_workers=args.n_prop_workers, descriptor_names=args.descriptor_names,
        n_desc_workers=args.n_desc_workers, traversal_layers=args.traversal_layers,
        n_pcs_traversal=args.n_pcs_traversal, n_steps=args.n_steps,
        pc_pct_lo=args.pc_percentile_lo, pc_pct_hi=args.pc_percentile_hi,
        n_background_scatter=args.n_background_scatter, device=args.device, seed=args.seed,
        refit=args.refit,
    )

    print(f"\nWrote {len(written)} artifacts to {data_dir(SLUG, args.tag)}")
    print(f"Draw the figures with:\n    python plotting/{SLUG}.py --tag {args.tag}")


if __name__ == "__main__":
    main()
