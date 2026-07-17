#!/usr/bin/env python
"""
build_cluster_llm_prompts.py
==============================

Builds a per-cluster CSV of prompts for later LLM-based cluster
labeling/annotation (e.g. "feed representative examples from each cluster
to Claude and ask it to name/describe them").

Consumes the outputs of this pipeline's earlier scripts rather than raw
paths, so it stays in sync with whatever checkpoint/dataset combination
was used upstream:

    - <prefix>_<split>_global_cond.pt   (from extract_cotrain_embeddings.py)
      -> smiles, dataset (source), ecfp
    - cotrain_cluster_labs.npy           (from cluster_cotrain_molecules.py)
      -> cluster label per molecule, aligned to the same concatenated
         smiles order used at clustering time
    - cluster_stats.csv                  (from cluster_cotrain_molecules.py)
      -> per-cluster descriptor means (MolWt, LogP, TPSA, Rings, n_molecules)

For each cluster (skipping clusters below `--min-cluster-size`), it picks:
    - up to `--n-representative` molecules nearest the cluster's ECFP
      centroid (Tanimoto distance to the mean fingerprint -- consistent
      with the Tanimoto-based clustering itself, rather than Euclidean),
    - up to `--n-diverse` molecules farthest from that centroid, to give
      the LLM a sense of the cluster's spread / outliers,

and assembles a free-text prompt block per cluster containing the SMILES
list, source-dataset breakdown, and the descriptor summary row, asking for:
    1. Short cluster name
    2. Shared structural themes
    3. Likely chemical class
    4. Confidence (0-100)

Output: cluster_llm_prompts.csv with columns [cluster, n_molecules, prompt].

Usage
-----
    python build_cluster_llm_prompts.py \\
        --data-dir /scratch/gpfs/ZHONGE/jc4587/nmr_embs_cotrain \\
        --prefix cotrain \\
        --splits train \\
        --cluster-dir cluster_summary \\
        --out-csv cluster_summary/cluster_llm_prompts.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Data loading (mirrors cluster_cotrain_molecules.py's loader, so ordering is
# guaranteed identical to what produced cotrain_cluster_labs.npy)
# -----------------------------------------------------------------------------


def load_cotrain_data(data_dir: Path, prefix: str, splits: Sequence[str]) -> Dict:
    smiles: List[str] = []
    dataset: List[str] = []
    split_tag: List[str] = []
    ecfp_parts: List[torch.Tensor] = []
    ecfp_radius = ecfp_nbits = None

    for split in splits:
        path = data_dir / f"{prefix}_{split}_global_cond.pt"
        if not path.exists():
            raise FileNotFoundError(f"Expected extraction output not found: {path}")
        d = torch.load(path, map_location="cpu")

        if ecfp_radius is None:
            ecfp_radius, ecfp_nbits = d["ecfp_radius"], d["ecfp_nbits"]

        n = len(d["smiles"])
        smiles.extend(d["smiles"])
        dataset.extend(d["dataset"])
        split_tag.extend([split] * n)
        ecfp_parts.append(d["ecfp"])

    ecfp = torch.cat(ecfp_parts, dim=0).numpy().astype(np.float32)
    return {
        "smiles": smiles, "dataset": dataset, "split": split_tag,
        "ecfp": ecfp, "ecfp_radius": ecfp_radius, "ecfp_nbits": ecfp_nbits,
    }


# -----------------------------------------------------------------------------
# Representative / diverse molecule selection
#
# Tanimoto distance to the cluster's mean fingerprint, rather than Euclidean
# norm as in the earlier ad-hoc script -- consistent with the Tanimoto/Butina
# clustering these labels came from. The "centroid" here is a soft mean
# fingerprint used purely to rank members by typicality; it is not treated as
# a molecule itself (only real cluster members are ever shown to the LLM).
# -----------------------------------------------------------------------------


def rank_by_typicality(cluster_ecfp: np.ndarray) -> np.ndarray:
    """Returns indices (local to this cluster) sorted from most to least
    typical, by Tanimoto similarity to the cluster's mean fingerprint."""
    centroid = cluster_ecfp.mean(axis=0, keepdims=True)  # [1, d], soft/fractional
    inter = (cluster_ecfp * centroid).sum(axis=1)
    union = cluster_ecfp.sum(axis=1) + centroid.sum() - inter
    sim = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    order = np.argsort(-sim)  # most similar (most typical) first
    return order


def select_examples(idx: np.ndarray, ecfp: np.ndarray, n_representative: int,
                     n_diverse: int) -> np.ndarray:
    cluster_ecfp = ecfp[idx]
    order = rank_by_typicality(cluster_ecfp)  # local order, most typical first

    rep_local = order[:n_representative]
    diverse_local = order[-n_diverse:] if n_diverse > 0 else np.array([], dtype=int)

    chosen_local = np.unique(np.concatenate([rep_local, diverse_local]))
    return idx[chosen_local]


# -----------------------------------------------------------------------------
# Prompt assembly
# -----------------------------------------------------------------------------


def build_prompt(cluster_id: int, mol_examples: List[str], dataset_counts: pd.Series,
                  stats_row: Optional[pd.Series]) -> str:
    mol_lines = "\n".join(f"- {s}" for s in mol_examples)
    source_lines = "\n".join(f"- {name}: {count}" for name, count in dataset_counts.items())

    if stats_row is not None:
        feature_lines = "\n".join(
            f"- {col}: {stats_row[col]:.3f}" if isinstance(stats_row[col], (int, float))
            else f"- {col}: {stats_row[col]}"
            for col in stats_row.index if col != "cluster"
        )
    else:
        feature_lines = "(no descriptor stats available for this cluster)"

    prompt = f"""Cluster {cluster_id}

Representative + diverse example molecules (SMILES):

{mol_lines}

Source dataset breakdown (of full cluster membership):

{source_lines}

Molecular feature summary (cluster-level means):

{feature_lines}

Please describe this cluster:
1. Short cluster name
2. Shared structural themes
3. Likely chemical class
4. Confidence (0-100)
"""
    return prompt


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a per-cluster CSV of LLM-labeling prompts from "
                     "cotrain extraction + clustering outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", type=Path, required=True,
                   help="Directory with <prefix>_<split>_global_cond.pt "
                        "(output of extract_cotrain_embeddings.py).")
    p.add_argument("--prefix", type=str, default="cotrain")
    p.add_argument("--splits", nargs="+", default=["train"],
                   choices=["train", "val", "test"],
                   help="Must match the splits used when cluster labels/stats "
                        "were generated.")
    p.add_argument("--cluster-dir", type=Path, required=True,
                   help="Directory containing cotrain_cluster_labs.npy and "
                        "cluster_stats.csv (output of cluster_cotrain_molecules.py).")
    p.add_argument("--cluster-labs-name", type=str, default="cotrain_cluster_labs.npy")
    p.add_argument("--cluster-stats-name", type=str, default="cluster_stats.csv")
    p.add_argument("--out-csv", type=Path, default=Path("cluster_summary/cluster_llm_prompts.csv"))
    p.add_argument("--min-cluster-size", type=int, default=5,
                   help="Skip clusters smaller than this (too few examples "
                        "for a meaningful prompt).")
    p.add_argument("--n-representative", type=int, default=8,
                   help="Most-typical examples per cluster (by Tanimoto "
                        "similarity to the cluster's mean fingerprint).")
    p.add_argument("--n-diverse", type=int, default=3,
                   help="Least-typical/outlier examples per cluster, to show "
                        "spread.")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading extracted data from {args.data_dir} (splits={args.splits}) ...")
    data = load_cotrain_data(args.data_dir, args.prefix, args.splits)
    smiles = data["smiles"]
    dataset_labels = pd.Series(data["dataset"])
    ecfp = data["ecfp"]
    n_total = len(smiles)

    labels_path = args.cluster_dir / args.cluster_labs_name
    labels = np.load(labels_path)
    if len(labels) != n_total:
        raise ValueError(
            f"Loaded {n_total} molecules from {args.data_dir} (splits={args.splits}) "
            f"but {labels_path} has {len(labels)} labels -- make sure --splits "
            f"matches what was used when cluster labels were generated."
        )
    print(f"Loaded {len(labels)} cluster labels from {labels_path}")

    stats_path = args.cluster_dir / args.cluster_stats_name
    cluster_stats = pd.read_csv(stats_path) if stats_path.exists() else None
    if cluster_stats is None:
        print(f"[warn] {stats_path} not found -- prompts will omit descriptor stats.")

    cluster_prompts = []
    unique_clusters, counts = np.unique(labels, return_counts=True)
    order = np.argsort(-counts)  # largest first, for a nicer output ordering

    for cluster_id in tqdm(unique_clusters[order], desc="Building cluster prompts"):
        idx = np.where(labels == cluster_id)[0]
        if len(idx) < args.min_cluster_size:
            continue

        chosen = select_examples(idx, ecfp, args.n_representative, args.n_diverse)
        mol_examples = [smiles[i] for i in chosen]

        dataset_counts = dataset_labels.iloc[idx].value_counts()

        stats_row = None
        if cluster_stats is not None:
            match = cluster_stats[cluster_stats["cluster"] == int(cluster_id)]
            if len(match) > 0:
                stats_row = match.iloc[0]

        prompt = build_prompt(int(cluster_id), mol_examples, dataset_counts, stats_row)

        cluster_prompts.append({
            "cluster": int(cluster_id),
            "n_molecules": int(len(idx)),
            "prompt": prompt,
        })

    df = pd.DataFrame(cluster_prompts)
    df.to_csv(args.out_csv, index=False)
    print(f"✓ Saved prompts for {len(df)} clusters to {args.out_csv}")


if __name__ == "__main__":
    main()
