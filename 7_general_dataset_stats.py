#!/usr/bin/env python
"""
general_dataset_stats.py
========================

Loads the cotrain extraction outputs, splits molecules by source dataset
(e.g. nmrexp, spectranp, uspto), computes a compact set of RDKit-based
molecular statistics, and saves both a CSV summary and several plots.

The plots are designed to match the style used elsewhere in this repository:
small, high-resolution Matplotlib figures with tight layout, simple axes,
and a clean panel-based layout for comparisons across datasets.

Usage
-----
python 7_general_dataset_stats.py \
    --data-dir /path/to/extracted_cotrain_files \
    --prefix cotrain \
    --splits train val test \
    --out-dir dataset_stats
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Optional, Sequence

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - environment dependent
    plt = None

try:
    import numpy as np
except ImportError:  # pragma: no cover - environment dependent
    np = None

try:
    import pandas as pd
except ImportError:  # pragma: no cover - environment dependent
    pd = None

try:
    import torch
except ImportError:  # pragma: no cover - environment dependent
    torch = None

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors
except ImportError:  # pragma: no cover - environment dependent
    Chem = None
    RDLogger = None
    Descriptors = None

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - environment dependent
    tqdm = None

if RDLogger is not None:
    RDLogger.DisableLog("rdApp.*")

DEFAULT_DESCRIPTORS = [
    "MolWt",
    "LogP",
    "TPSA",
    "RingCount",
    "NumRotatableBonds",
    "NumHDonors",
    "NumHAcceptors",
]

DPI = 600
FIGSIZE = (10, 10)


def load_cotrain_data(data_dir: Path, prefix: str, splits: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for split in splits:
        path = data_dir / f"{prefix}_{split}_global_cond.pt"
        if not path.exists():
            raise FileNotFoundError(f"Expected extraction output not found: {path}")

        print(f"Loading {path} ...")
        d = torch.load(path, map_location="cpu")

        smiles = list(d["smiles"])
        dataset_labels = list(map(str, d["dataset"]))
        if len(smiles) != len(dataset_labels):
            raise ValueError(f"Length mismatch in {path}: {len(smiles)} smiles vs {len(dataset_labels)} dataset labels")

        for smi, ds in zip(smiles, dataset_labels):
            rows.append({"split": split, "dataset": ds, "smiles": smi})

    return pd.DataFrame(rows)


def _descriptor_worker(smi: str) -> Dict[str, float]:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return {name: np.nan for name in DEFAULT_DESCRIPTORS}

    vals: Dict[str, float] = {}
    try:
        vals["MolWt"] = float(Descriptors.MolWt(mol))
        vals["LogP"] = float(Descriptors.MolLogP(mol))
        vals["TPSA"] = float(Descriptors.TPSA(mol))
        vals["RingCount"] = float(Descriptors.RingCount(mol))
        vals["NumRotatableBonds"] = float(Descriptors.NumRotatableBonds(mol))
        vals["NumHDonors"] = float(Descriptors.NumHDonors(mol))
        vals["NumHAcceptors"] = float(Descriptors.NumHAcceptors(mol))
    except Exception:
        return {name: np.nan for name in DEFAULT_DESCRIPTORS}
    return vals


def compute_descriptor_table(smiles_list: Sequence[str], n_workers: int) -> pd.DataFrame:
    if n_workers <= 1:
        rows = [_descriptor_worker(s) for s in tqdm(smiles_list, desc="Computing RDKit descriptors")]
    else:
        with mp.Pool(n_workers) as pool:
            rows = list(
                tqdm(
                    pool.imap(_descriptor_worker, smiles_list, chunksize=256),
                    total=len(smiles_list),
                    desc=f"Computing RDKit descriptors ({n_workers} workers)",
                )
            )

    return pd.DataFrame(rows, columns=DEFAULT_DESCRIPTORS)


def summarize_by_dataset(df: pd.DataFrame, n_workers: int) -> pd.DataFrame:
    desc_df = compute_descriptor_table(df["smiles"].tolist(), n_workers=n_workers)
    out_df = pd.concat([df.reset_index(drop=True), desc_df], axis=1)

    out_df["valid_smiles"] = out_df["MolWt"].notna()

    summary_rows: List[Dict[str, object]] = []
    for dataset_name, group in out_df.groupby("dataset", sort=True):
        group = group.copy()
        summary_rows.append(
            {
                "dataset": dataset_name,
                "n_molecules": int(len(group)),
                "n_valid_smiles": int(group["valid_smiles"].sum()),
                "valid_smiles_fraction": float(group["valid_smiles"].mean()),
            }
        )
        for desc in DEFAULT_DESCRIPTORS:
            valid = group[desc].notna()
            if valid.sum() == 0:
                summary_rows[-1][f"mean_{desc}"] = np.nan
                summary_rows[-1][f"std_{desc}"] = np.nan
                summary_rows[-1][f"median_{desc}"] = np.nan
            else:
                vals = group.loc[valid, desc]
                summary_rows[-1][f"mean_{desc}"] = float(vals.mean())
                summary_rows[-1][f"std_{desc}"] = float(vals.std())
                summary_rows[-1][f"median_{desc}"] = float(vals.median())

    return pd.DataFrame(summary_rows)


def plot_dataset_counts(summary_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(summary_df))
    ax.bar(x, summary_df["n_molecules"], color="#4C78A8", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["dataset"], rotation=20, ha="right")
    ax.set_ylabel("n molecules")
    ax.set_title("Cotrain dataset sizes")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved {out_path}")


def plot_descriptor_boxplots(stats_df: pd.DataFrame, out_path: Path) -> None:
    datasets = sorted(stats_df["dataset"].unique())
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), squeeze=False)

    for ax, desc in zip(axes.flat, DEFAULT_DESCRIPTORS):
        data = []
        labels = []
        for ds in datasets:
            vals = stats_df.loc[stats_df["dataset"] == ds, desc].dropna().to_numpy()
            if len(vals) == 0:
                continue
            data.append(vals)
            labels.append(ds)

        if len(data) == 0:
            ax.axis("off")
            continue

        bp = ax.boxplot(data, patch_artist=True, labels=labels)
        for box in bp["boxes"]:
            box.set(facecolor="#9ECae1", alpha=0.8)
        for whisker in bp["whiskers"]:
            whisker.set(color="#4C78A8", linewidth=1.0)
        for cap in bp["caps"]:
            cap.set(color="#4C78A8", linewidth=1.0)
        for median in bp["medians"]:
            median.set(color="#1F4E79", linewidth=1.2)

        ax.set_title(desc)
        ax.set_ylabel(desc)
        ax.grid(True, axis="y", alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Hide the final unused panel if needed
    for ax in axes.flat[len(DEFAULT_DESCRIPTORS):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved {out_path}")


def plot_descriptor_histograms(stats_df: pd.DataFrame, out_path: Path) -> None:
    datasets = sorted(stats_df["dataset"].unique())
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), squeeze=False)

    for ax, desc in zip(axes.flat, DEFAULT_DESCRIPTORS):
        for ds in datasets:
            vals = stats_df.loc[stats_df["dataset"] == ds, desc].dropna().to_numpy()
            if len(vals) == 0:
                continue
            ax.hist(vals, bins=30, alpha=0.35, label=ds, linewidth=0)
        ax.set_title(desc)
        ax.set_xlabel(desc)
        ax.set_ylabel("count")
        ax.grid(True, alpha=0.15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Avoid duplicate legends on every panel by adding one only to the last panel
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=4, frameon=False)

    for ax in axes.flat[len(DEFAULT_DESCRIPTORS):]:
        ax.axis("off")

    plt.tight_layout(rect=(0, 0.05, 1, 1))
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved {out_path}")


def ensure_dependencies() -> None:
    missing = []
    if np is None:
        missing.append("numpy")
    if pd is None:
        missing.append("pandas")
    if torch is None:
        missing.append("torch")
    if plt is None:
        missing.append("matplotlib")
    if Chem is None or Descriptors is None:
        missing.append("rdkit")
    if tqdm is None:
        missing.append("tqdm")
    if missing:
        raise ImportError(f"Missing required dependencies: {', '.join(missing)}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute and plot general molecular statistics for each source dataset in the cotrain corpus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", type=Path, required=True,
                   help="Directory containing <prefix>_<split>_global_cond.pt files.")
    p.add_argument("--prefix", type=str, default="cotrain")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                   choices=["train", "val", "test"])
    p.add_argument("--datasets", nargs="+", default=None,
                   help="Optional subset of source datasets to include (for example nmrexp spectranp uspto).")
    p.add_argument("--out-dir", type=Path, default=Path("dataset_stats"))
    p.add_argument("--n-workers", type=int, default=max(1, mp.cpu_count() - 2))
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    global args
    args = parse_args(argv)
    ensure_dependencies()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("General dataset statistics for cotrain")
    print(f"  data dir : {args.data_dir}")
    print(f"  prefix   : {args.prefix}")
    print(f"  splits   : {args.splits}")
    print("=" * 78)

    df = load_cotrain_data(args.data_dir, args.prefix, args.splits)
    if args.datasets is not None:
        datasets = set(args.datasets)
        df = df[df["dataset"].isin(datasets)].copy()

    if df.empty:
        raise ValueError("No molecules were loaded from the requested data files.")

    descriptor_table = compute_descriptor_table(df["smiles"].tolist(), n_workers=args.n_workers)
    stats_df = pd.concat([df.reset_index(drop=True), descriptor_table], axis=1)
    summary_df = summarize_by_dataset(df, n_workers=args.n_workers)
    summary_df.to_csv(args.out_dir / f"{args.prefix}_dataset_stats.csv", index=False)
    print(f"✓ Saved {args.out_dir / f'{args.prefix}_dataset_stats.csv'}")

    plot_dataset_counts(summary_df, args.out_dir / f"{args.prefix}_dataset_counts.png")
    plot_descriptor_boxplots(stats_df, args.out_dir / f"{args.prefix}_descriptor_boxplots.png")
    plot_descriptor_histograms(stats_df, args.out_dir / f"{args.prefix}_descriptor_histograms.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
