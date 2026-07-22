#!/usr/bin/env python
"""
11_make_pretty_plots.py
==========================

Driver script that runs 4_pretty_plot.py six times to produce the full
set of "pretty plot" UMAP figures for the cotrain dataset:

    embedding in {ecfp, global_cond}
        x
    color-by in {dataset, np_class, sim_real}

For each embedding type, the 2D UMAP projection is computed ONCE and cached
(--save-umap-embedding on the first call), then reused for the other two
colorings of that same embedding (--umap-embedding-path) -- so all three
colorings of e.g. "ecfp" share the exact same point layout and are directly
comparable, and you only pay for UMAP's expensive fit step twice total
(once per embedding type) instead of six times.

Note on metrics: ECFP fingerprints are binary, so this script defaults to
--umap-metric jaccard for the ecfp runs (the standard similarity metric for
binary fingerprints) rather than the cosine default used for the continuous
global_cond embedding. Override either via --ecfp-metric / --global-cond-metric
if you disagree.

Requires
--------
    - `<prefix>_<split>_global_cond.pt` files (as used by 4_pretty_plot.py)
    - `labels.pkl` (from npclassifier_local.py) for the np_class colorings
    - 4_pretty_plot.py in the same directory (or on PYTHONPATH)

Usage
-----
python 4a_make_pretty_plots.py --data-dir /scratch/gpfs/ZHONGE/jc4587/nmr_embs_FINAL --np-class-labels-path /scratch/gpfs/ZHONGE/jc4587/nmr_embs_FINAL/labels.pkl --out-dir 4a_make_pretty_plots

Output layout
-------------
    pretty_plots/
        ecfp_dataset/umap_ecfp_dataset.png
        ecfp_np_class/umap_ecfp_np_class.png
        ecfp_sim_real/umap_ecfp_sim_real.png
        global_cond_dataset/umap_global_cond_dataset.png
        global_cond_np_class/umap_global_cond_np_class.png
        global_cond_sim_real/umap_global_cond_sim_real.png
        umap_cache_ecfp.npy         (cached 2D projection, reused across the 3 ecfp plots)
        umap_cache_global_cond.npy  (cached 2D projection, reused across the 3 global_cond plots)
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence


EMBEDDING_KEYS = ['global_cond'] # ["ecfp", "global_cond"]
COLOR_MODES = ["dataset", "np_class", "sim_real"]

DEFAULT_METRIC_BY_EMBEDDING = {
    "ecfp": "jaccard",
    "global_cond": "cosine",
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run 4_pretty_plot.py for {ecfp, global_cond} x "
                    "{dataset, np_class, sim_real} -- 6 figures total.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--prefix", type=str, default="cotrain")
    p.add_argument("--splits", nargs="+", default=["train"], choices=["train", "val", "test"])
    p.add_argument("--datasets", nargs="+", default=None,
                   help="Restrict to these source datasets. Default: ALL.")

    p.add_argument("--np-class-labels-path", type=Path, required=True,
                   help="labels.pkl from npclassifier_local.py (required for the "
                        "np_class colorings; must be aligned to --splits/--datasets).")
    p.add_argument("--np-class-probs-path", type=Path, default=None,
                   help="labels_probs.pkl from npclassifier_local.py. If not given, "
                        "defaults to --np-class-labels-path with '_probs' inserted "
                        "before the extension (e.g. labels.pkl -> labels_probs.pkl).")
    p.add_argument("--np-class-min-probability", type=float, default=0.99,
                   help="Molecules below this confidence are grayed out ('Other') "
                        "instead of colored by class.")
    p.add_argument("--real-prefixes", nargs="+", default=["real"])
    p.add_argument("--sim-prefixes", nargs="+", default=["syn"])
    p.add_argument("--np-class-top-n", type=int, default=10)

    p.add_argument("--ecfp-metric", type=str, default=DEFAULT_METRIC_BY_EMBEDDING["ecfp"])
    p.add_argument("--global-cond-metric", type=str, default=DEFAULT_METRIC_BY_EMBEDDING["global_cond"])
    p.add_argument("--umap-n-neighbors", type=int, default=30)
    p.add_argument("--umap-min-dist", type=float, default=0.1)

    p.add_argument("--umap-script", type=Path, default=Path(__file__).parent / "4_pretty_plot.py",
                   help="Path to 4_pretty_plot.py")
    p.add_argument("--out-dir", type=Path, default=Path("pretty_plots"))
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true",
                   help="Print the 6 commands that would be run, without running them.")
    return p.parse_args(argv)


def build_common_args(args: argparse.Namespace) -> List[str]:
    common = [
        "--data-dir", str(args.data_dir),
        "--prefix", args.prefix,
        "--splits", *args.splits,
        "--umap-n-neighbors", str(args.umap_n_neighbors),
        "--umap-min-dist", str(args.umap_min_dist),
        "--dpi", str(args.dpi),
        "--seed", str(args.seed),
    ]
    if args.datasets:
        common += ["--datasets", *args.datasets]
    return common


def run_one(args: argparse.Namespace, embedding_key: str, color_by: str,
            metric: str, umap_cache_path: Path, is_first_for_embedding: bool) -> None:
    out_dir = args.out_dir / f"{embedding_key}_{color_by}"
    cmd = [
        sys.executable, str(args.umap_script),
        *build_common_args(args),
        "--embedding-key", embedding_key,
        "--umap-metric", metric,
        "--color-by", color_by,
        "--out-dir", str(out_dir),
        "--main-fig-name", f"umap_{embedding_key}_{color_by}.png",
    ]

    if color_by == "np_class":
        probs_path = args.np_class_probs_path
        if probs_path is None:
            probs_path = args.np_class_labels_path.with_name(
                args.np_class_labels_path.stem + "_probs" + args.np_class_labels_path.suffix
            )
        cmd += [
            "--np-class-labels-path", str(args.np_class_labels_path),
            "--np-class-probs-path", str(probs_path),
            "--np-class-min-probability", str(args.np_class_min_probability),
            "--np-class-top-n", str(args.np_class_top_n),
        ]
    elif color_by == "sim_real":
        cmd += [
            "--real-prefixes", *args.real_prefixes,
            "--sim-prefixes", *args.sim_prefixes,
        ]

    # Reuse the same 2D projection across all 3 colorings of this embedding:
    # the first run for a given embedding_key computes + caches it, the rest
    # just load the cached one.
    if is_first_for_embedding:
        cmd += ["--save-umap-embedding", str(umap_cache_path)]
    else:
        cmd += ["--umap-embedding-path", str(umap_cache_path)]

    print("\n" + "=" * 78)
    print(f"[{embedding_key} / {color_by}]")
    print(" ".join(cmd))
    print("=" * 78)

    if args.dry_run:
        return

    subprocess.run(cmd, check=True)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    metric_by_embedding = {
        "ecfp": args.ecfp_metric,
        "global_cond": args.global_cond_metric,
    }

    for embedding_key in EMBEDDING_KEYS:
        umap_cache_path = args.out_dir / f"umap_cache_{embedding_key}.npy"
        for i, color_by in enumerate(COLOR_MODES):
            run_one(
                args, embedding_key, color_by,
                metric=metric_by_embedding[embedding_key],
                umap_cache_path=umap_cache_path,
                is_first_for_embedding=(i == 0),
            )

    print("\nAll 6 figures done." if not args.dry_run else "\nDry run complete (no commands executed).")


if __name__ == "__main__":
    main()

# #!/usr/bin/env python
# """
# 11_make_pretty_plots.py
# ==========================

# Driver script that runs 4_pretty_plot.py six times to produce the full
# set of "pretty plot" UMAP figures for the cotrain dataset:

#     embedding in {ecfp, global_cond}
#         x
#     color-by in {dataset, np_class, sim_real}

# For each embedding type, the 2D UMAP projection is computed ONCE and cached
# (--save-umap-embedding on the first call), then reused for the other two
# colorings of that same embedding (--umap-embedding-path) -- so all three
# colorings of e.g. "ecfp" share the exact same point layout and are directly
# comparable, and you only pay for UMAP's expensive fit step twice total
# (once per embedding type) instead of six times.

# Note on metrics: ECFP fingerprints are binary, so this script defaults to
# --umap-metric jaccard for the ecfp runs (the standard similarity metric for
# binary fingerprints) rather than the cosine default used for the continuous
# global_cond embedding. Override either via --ecfp-metric / --global-cond-metric
# if you disagree.

# Requires
# --------
#     - `<prefix>_<split>_global_cond.pt` files (as used by 4_pretty_plot.py)
#     - `labels.pkl` (from npclassifier_local.py) for the np_class colorings
#     - 4_pretty_plot.py in the same directory (or on PYTHONPATH)

# Usage
# -----
# python 4a_make_pretty_plots.py --data-dir /scratch/gpfs/ZHONGE/jc4587/nmr_embs_FINAL --np-class-labels-path /scratch/gpfs/ZHONGE/jc4587/nmr_embs_FINAL/labels.pkl --out-dir 4a_make_pretty_plots

# Output layout
# -------------
#     pretty_plots/
#         ecfp_dataset/umap_ecfp_dataset.png
#         ecfp_np_class/umap_ecfp_np_class.png
#         ecfp_sim_real/umap_ecfp_sim_real.png
#         global_cond_dataset/umap_global_cond_dataset.png
#         global_cond_np_class/umap_global_cond_np_class.png
#         global_cond_sim_real/umap_global_cond_sim_real.png
#         umap_cache_ecfp.npy         (cached 2D projection, reused across the 3 ecfp plots)
#         umap_cache_global_cond.npy  (cached 2D projection, reused across the 3 global_cond plots)
# """

# import argparse
# import subprocess
# import sys
# from pathlib import Path
# from typing import List, Optional, Sequence


# EMBEDDING_KEYS = ["ecfp", "global_cond"]
# COLOR_MODES = ["dataset", "np_class", "sim_real"]

# DEFAULT_METRIC_BY_EMBEDDING = {
#     "ecfp": "jaccard",
#     "global_cond": "cosine",
# }


# def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
#     p = argparse.ArgumentParser(
#         description="Run 4_pretty_plot.py for {ecfp, global_cond} x "
#                     "{dataset, np_class, sim_real} -- 6 figures total.",
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter,
#     )
#     p.add_argument("--data-dir", type=Path, required=True)
#     p.add_argument("--prefix", type=str, default="cotrain")
#     p.add_argument("--splits", nargs="+", default=["train"], choices=["train", "val", "test"])
#     p.add_argument("--datasets", nargs="+", default=None,
#                    help="Restrict to these source datasets. Default: ALL.")

#     p.add_argument("--np-class-labels-path", type=Path, required=True,
#                    help="labels.pkl from npclassifier_local.py (required for the "
#                         "np_class colorings; must be aligned to --splits/--datasets).")
#     p.add_argument("--real-prefixes", nargs="+", default=["real"])
#     p.add_argument("--sim-prefixes", nargs="+", default=["syn"])
#     p.add_argument("--np-class-top-n", type=int, default=10)

#     p.add_argument("--ecfp-metric", type=str, default=DEFAULT_METRIC_BY_EMBEDDING["ecfp"])
#     p.add_argument("--global-cond-metric", type=str, default=DEFAULT_METRIC_BY_EMBEDDING["global_cond"])
#     p.add_argument("--umap-n-neighbors", type=int, default=30)
#     p.add_argument("--umap-min-dist", type=float, default=0.1)

#     p.add_argument("--umap-script", type=Path, default=Path(__file__).parent / "4_pretty_plot.py",
#                    help="Path to 4_pretty_plot.py")
#     p.add_argument("--out-dir", type=Path, default=Path("pretty_plots"))
#     p.add_argument("--dpi", type=int, default=600)
#     p.add_argument("--seed", type=int, default=0)
#     p.add_argument("--dry-run", action="store_true",
#                    help="Print the 6 commands that would be run, without running them.")
#     return p.parse_args(argv)


# def build_common_args(args: argparse.Namespace) -> List[str]:
#     common = [
#         "--data-dir", str(args.data_dir),
#         "--prefix", args.prefix,
#         "--splits", *args.splits,
#         "--umap-n-neighbors", str(args.umap_n_neighbors),
#         "--umap-min-dist", str(args.umap_min_dist),
#         "--dpi", str(args.dpi),
#         "--seed", str(args.seed),
#     ]
#     if args.datasets:
#         common += ["--datasets", *args.datasets]
#     return common


# def run_one(args: argparse.Namespace, embedding_key: str, color_by: str,
#             metric: str, umap_cache_path: Path, is_first_for_embedding: bool) -> None:
#     out_dir = args.out_dir / f"{embedding_key}_{color_by}"
#     cmd = [
#         sys.executable, str(args.umap_script),
#         *build_common_args(args),
#         "--embedding-key", embedding_key,
#         "--umap-metric", metric,
#         "--color-by", color_by,
#         "--out-dir", str(out_dir),
#         "--main-fig-name", f"umap_{embedding_key}_{color_by}.png",
#     ]

#     if color_by == "np_class":
#         cmd += [
#             "--np-class-labels-path", str(args.np_class_labels_path),
#             "--np-class-top-n", str(args.np_class_top_n),
#         ]
#     elif color_by == "sim_real":
#         cmd += [
#             "--real-prefixes", *args.real_prefixes,
#             "--sim-prefixes", *args.sim_prefixes,
#         ]

#     # Reuse the same 2D projection across all 3 colorings of this embedding:
#     # the first run for a given embedding_key computes + caches it, the rest
#     # just load the cached one.
#     if is_first_for_embedding:
#         cmd += ["--save-umap-embedding", str(umap_cache_path)]
#     else:
#         cmd += ["--umap-embedding-path", str(umap_cache_path)]

#     print("\n" + "=" * 78)
#     print(f"[{embedding_key} / {color_by}]")
#     print(" ".join(cmd))
#     print("=" * 78)

#     if args.dry_run:
#         return

#     subprocess.run(cmd, check=True)


# def main(argv: Optional[Sequence[str]] = None) -> None:
#     args = parse_args(argv)
#     args.out_dir.mkdir(parents=True, exist_ok=True)

#     metric_by_embedding = {
#         "ecfp": args.ecfp_metric,
#         "global_cond": args.global_cond_metric,
#     }

#     for embedding_key in EMBEDDING_KEYS:
#         umap_cache_path = args.out_dir / f"umap_cache_{embedding_key}.npy"
#         for i, color_by in enumerate(COLOR_MODES):
#             run_one(
#                 args, embedding_key, color_by,
#                 metric=metric_by_embedding[embedding_key],
#                 umap_cache_path=umap_cache_path,
#                 is_first_for_embedding=(i == 0),
#             )

#     print("\nAll 6 figures done." if not args.dry_run else "\nDry run complete (no commands executed).")


# if __name__ == "__main__":
#     main()
