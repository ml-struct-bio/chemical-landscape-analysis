#!/usr/bin/env python
"""
06_dataset_stats.py
====================

Per-dataset characterization and comparison, molecular terms only: the
34-descriptor RDKit panel plus element composition, ring topology,
stereochemistry, Murcko scaffold diversity and exact cross-dataset molecule
overlap, for every source in the corpus.

Port of the previous pipeline's `9_run_dataset_stats_experiment.py`. Same
algorithm, same defaults, same numbers. Its spectral half (`--spectral`, live
peak re-extraction via a checkpoint/GPU) is NOT ported -- it depends on script
17's extraction machinery, which does not exist in this repo yet.

Outputs (`data/06_dataset_stats/<tag>/`)
-----------------------------------------
    stats.csv               per-molecule sampled table: descriptors, composition,
                             rings, stereochemistry, Murcko scaffold
    counts.csv               exact dataset x split molecule counts, full corpus
    summary.csv               per-dataset mean/std/median of every feature
    divergence.csv             per-feature KS statistic + standardized mean
                               difference, every dataset pair
    overlap_pairs.csv          exact cross-dataset molecule sharing (canonical
                               SMILES, full corpus)
    overlap_summary.csv        per-dataset unique count, internal duplicate rate,
                               fraction shared with another source
    scaffold_summary.csv       per-dataset scaffold count/diversity/concentration
    scaffold_top.csv           each dataset's most common Murcko scaffolds
    scaffold_coverage.csv      scaffold-rank coverage curves (long form)

Usage
-----
    python analysis/06_dataset_stats.py \
        --data-dir /scratch/.../26-07-27-cotrain-v3-dedupOFF-s0/epoch1399 \
        --prefix cotrain --splits train --max-per-dataset 200000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.analysis.dataset_stats import run_dataset_stats  # noqa: E402
from src.common.paths import data_dir  # noqa: E402
from src.common.workers import safe_n_workers  # noqa: E402

SLUG = "06_dataset_stats"


def parse_args():
    p = argparse.ArgumentParser(
        description="Characterize and compare the corpus's source datasets, molecular terms only.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-dir", type=Path, required=True,
                   help="Directory holding extraction's <prefix>_<split>_global_cond.pt.")
    p.add_argument("--prefix", type=str, default="cotrain")
    p.add_argument("--splits", nargs="+", default=["train"], choices=["train", "val", "test"])
    p.add_argument("--datasets", nargs="+", default=None,
                   help="Restrict the analysis to these source datasets.")
    p.add_argument("--tag", type=str, default="main",
                   help="Output tag, i.e. data/06_dataset_stats/<tag>/.")

    p.add_argument("--max-per-dataset", type=int, default=200_000,
                   help="Cap per dataset for the EXPENSIVE per-molecule pass (34 "
                        "descriptors, stereocenters, Murcko scaffolds). 0 = no cap. "
                        "Dataset sizes and cross-dataset overlap are always computed "
                        "exactly over the full corpus regardless.")
    p.add_argument("--top-k-scaffolds", type=int, default=15,
                   help="How many of each dataset's most common Murcko scaffolds to "
                        "record in scaffold_top.csv.")
    p.add_argument("--n-workers", type=int, default=safe_n_workers())
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 78)
    print("Dataset statistics (molecular)")
    print(f"  data dir          : {args.data_dir}")
    print(f"  prefix/splits     : {args.prefix} / {args.splits}")
    print(f"  max per dataset   : {args.max_per_dataset or '(no cap)'}")
    print(f"  tag               : {args.tag}")
    print("=" * 78)

    written = run_dataset_stats(
        data_dir=args.data_dir, prefix=args.prefix, splits=args.splits,
        out_dir=data_dir(SLUG, args.tag, create=True), tag=args.tag, slug=SLUG,
        datasets=args.datasets, n_workers=args.n_workers,
        max_per_dataset=args.max_per_dataset or None,
        top_k_scaffolds=args.top_k_scaffolds, seed=args.seed,
    )

    print(f"\nWrote {len(written)} artifacts to {data_dir(SLUG, args.tag)}")
    print(f"Draw the figures with:\n    python plotting/{SLUG}.py --tag {args.tag}")


if __name__ == "__main__":
    main()
