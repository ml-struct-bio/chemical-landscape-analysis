#!/usr/bin/env python
"""
03_clustering.py
================

Butina clustering of the cotrain corpus over ECFP fingerprints.

Butina is exact and quadratic, so it is run on a **stratified subsample**
(proportional per source dataset), and every molecule in the full corpus is then
assigned to its nearest cluster prototype by Tanimoto similarity. The subsampled
molecules keep their exact Butina label rather than a nearest-prototype
approximation of it.

Port of the previous pipeline's `2_run_clustering_experiment.py`. Same
algorithm, same defaults, same cluster labels.

The Butina step is cached under `cache/butina/`, keyed on corpus size, source
composition, subsample size, cutoff, seed and ECFP settings -- so re-running with
different plotting or descriptor options does not redo the expensive part. This
replaces the old `--load-prototypes` hand-managed pickle; `--refit` forces it.

Outputs (`data/03_clustering/<tag>/`)
------------------------------------
    cluster_labels.npy      int32, one label per molecule, in extraction order
    descriptors.npz         MolWt/LogP/TPSA/Rings per molecule
    cluster_stats.csv       per cluster: size + mean descriptors, largest first
    plot_clusters.csv       the top-N clusters the figure draws, with normalized bars
    representatives.csv     the sampled SMILES per plotted cluster
    cluster_meta.csv        per-molecule flat table (only with --write-meta-csv)

Needs a GPU only to go faster: `--device cuda` accelerates the pairwise Tanimoto
and the assignment pass, but neither needs a checkpoint.

Usage
-----
    python analysis/03_clustering.py \
        --data-dir /scratch/.../26-07-27-cotrain-v3-dedupOFF-s0/epoch1399 \
        --prefix cotrain --splits train \
        --n-cluster-sample 100000 --butina-cutoff 0.35
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402

from src.analysis.clustering import run_clustering  # noqa: E402
from src.common.paths import data_dir  # noqa: E402
from src.common.workers import safe_n_workers  # noqa: E402

SLUG = "03_clustering"


def parse_args():
    p = argparse.ArgumentParser(
        description="Butina-cluster the corpus over ECFP fingerprints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-dir", type=Path, required=True,
                   help="Directory holding extraction's <prefix>_<split>_global_cond.pt.")
    p.add_argument("--prefix", type=str, default="cotrain")
    p.add_argument("--splits", nargs="+", default=["train"], choices=["train", "val", "test"])
    p.add_argument("--tag", type=str, default="main",
                   help="Output tag, i.e. data/03_clustering/<tag>/.")

    p.add_argument("--n-cluster-sample", type=int, default=10_000,
                   help="Molecules drawn (stratified by source dataset) for the exact "
                        "Butina pass. Cost is quadratic in this.")
    p.add_argument("--butina-cutoff", type=float, default=0.35,
                   help="Tanimoto DISTANCE cutoff; lower means tighter, more numerous "
                        "clusters.")
    p.add_argument("--assign-chunk-size", type=int, default=50_000,
                   help="Molecules per chunk in the nearest-prototype assignment pass.")
    p.add_argument("--refit", action="store_true",
                   help="Ignore any cached Butina prototypes and re-run the clustering.")
    p.add_argument("--load-prototypes", type=Path, default=None,
                   help="Use this specific prototype pickle instead of the keyed cache "
                        "entry. Mainly for reproducing an older clustering.")

    p.add_argument("--max-clusters-plot", type=int, default=24,
                   help="How many of the largest clusters the figure gets data for.")
    p.add_argument("--n-reps", type=int, default=6,
                   help="Representative molecules sampled per plotted cluster.")

    p.add_argument("--n-desc-workers", type=int, default=safe_n_workers())
    p.add_argument("--no-descriptor-cache", action="store_true",
                   help="Recompute descriptors instead of reusing cache/descriptors/.")
    p.add_argument("--write-meta-csv", action="store_true",
                   help="Also write the old pipeline's flat per-molecule cluster_meta.csv. "
                        "Off by default: at full corpus scale it is a ~500 MB file whose "
                        "only new columns are already saved compactly as "
                        "cluster_labels.npy + descriptors.npz.")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 78)
    print("Butina clustering (Tanimoto over ECFP)")
    print(f"  data dir          : {args.data_dir}")
    print(f"  prefix/splits     : {args.prefix} / {args.splits}")
    print(f"  cluster subsample : {args.n_cluster_sample}")
    print(f"  butina cutoff     : {args.butina_cutoff}")
    print(f"  device            : {args.device}")
    print(f"  tag               : {args.tag}")
    print("=" * 78)

    written = run_clustering(
        data_dir=args.data_dir, prefix=args.prefix, splits=args.splits,
        out_dir=data_dir(SLUG, args.tag, create=True), tag=args.tag, slug=SLUG,
        n_cluster_sample=args.n_cluster_sample, butina_cutoff=args.butina_cutoff,
        max_clusters_plot=args.max_clusters_plot, n_reps=args.n_reps,
        assign_chunk_size=args.assign_chunk_size, n_desc_workers=args.n_desc_workers,
        device=args.device, seed=args.seed, refit=args.refit,
        load_prototypes=args.load_prototypes, write_meta_csv=args.write_meta_csv,
        use_descriptor_cache=not args.no_descriptor_cache,
    )

    print(f"\nWrote {len(written)} artifacts to {data_dir(SLUG, args.tag)}")
    print(f"Draw the figures with:\n    python plotting/{SLUG}.py --tag {args.tag}")


if __name__ == "__main__":
    main()
