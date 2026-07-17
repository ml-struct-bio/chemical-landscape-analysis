#!/usr/bin/env python
"""
compare_spectranp_nmrexp.py
==============================

Compares the chemical space of SpectraNP vs. NMRExp using ONLY pre-extracted
data -- no live Hydra config / `NMRDataModule` / dataloader dependency at
all. This sidesteps the broken `nmrexp-nmrpeak-recon-fullh` /
`_or_recon_heavy` split-suffix path entirely (that split index was never
built for this checkout, per the KeyError from the live-loading version);
everything here comes straight from what `extract_cotrain_embeddings.py`
already saved to disk.

Reads either:
    - a single merged payload (e.g. `cotrain_train_global_cond.pt`),
      filtered to each source via its saved `"dataset"` column, or
    - two separate per-source payloads (e.g. if you ran the extraction
      script with `--save-per-source`, giving you
      `spectranp_train_global_cond.pt` / `nmrexp_train_global_cond.pt`
      directly).

Reuses the `"ecfp"` field already computed at extraction time by default
(no redundant fingerprinting) -- pass `--force-recompute-ecfp` if you want
a different radius/nBits than what was saved. `mol_idx` is carried through
into the summary CSV whenever the source payload has it.

Usage
-----
    # Single merged cotrain payload, filtered by the 'dataset' column
python 8_compare_spectranp_nmrexp_full.py --data-path /scratch/gpfs/ZHONGE/jc4587/nmr_embs_FINAL/cotrain_train_global_cond.pt --out-dir spectranp_nmrexp_comparison

    # Separate per-source payloads
    python compare_spectranp_nmrexp.py \\
        --spectranp-data spectranp_train_global_cond.pt \\
        --nmrexp-data nmrexp_train_global_cond.pt \\
        --out-dir spectranp_nmrexp_comparison
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from sklearn.decomposition import PCA
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")

DEFAULT_DESCRIPTORS = [
    "MolWt",
    "ExactMolWt",
    "LogP",
    "TPSA",
    "HeavyAtomCount",
    "NumAtoms",
    "NumBonds",
    "NumHeteroatoms",
    "RingCount",
    "NumAromaticRings",
    "NumAliphaticRings",
    "NumSaturatedRings",
    "NumRotatableBonds",
    "NumHDonors",
    "NumHAcceptors",
    "FormalCharge",
    "FractionCSP3",
    "NumAmideBonds",
    "NumBridgeheadAtoms",
    "NumSpiroAtoms",
]

QUALITATIVE_PALETTE = plt.cm.tab10(np.linspace(0, 1, 10))
DPI = 600


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare SpectraNP and NMRExp chemical space using only "
                     "pre-extracted embedding/ECFP payloads.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-path", type=Path, default=None,
                   help="A single merged payload (e.g. cotrain_train_global_cond.pt) "
                        "to pull BOTH sources from, filtered by its saved "
                        "'dataset' column. Ignored for a source if that "
                        "source's --*-data override is given.")
    p.add_argument("--spectranp-data", type=Path, default=None,
                   help="Override: payload to use for the spectranp source "
                        "(e.g. a --save-per-source spectranp_*.pt file). "
                        "Defaults to --data-path filtered to dataset=='spectranp'.")
    p.add_argument("--nmrexp-data", type=Path, default=None,
                   help="Override: payload to use for the nmrexp source. "
                        "Defaults to --data-path filtered to dataset=='nmrexp'.")
    p.add_argument("--out-dir", type=Path, default=Path("spectranp_nmrexp_comparison"))
    p.add_argument("--max-samples", type=int, default=None,
                   help="Max molecules drawn from each source (random subsample). Leave as None to use the full dataset.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force-recompute-ecfp", action="store_true",
                   help="Recompute ECFPs instead of reusing the saved 'ecfp' "
                        "field (only needed if you want a different "
                        "radius/nBits than what was originally extracted).")
    p.add_argument("--fps-radius", type=int, default=2)
    p.add_argument("--fps-bits", type=int, default=2048)
    p.add_argument("--n-workers", type=int, default=max(1, mp.cpu_count() - 2))
    return p.parse_args(argv)


# -----------------------------------------------------------------------------
# Loading (pre-extracted payloads only)
# -----------------------------------------------------------------------------


def load_payload(path: Path, expected_source: Optional[str] = None) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Input payload not found: {path}")

    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict) or "smiles" not in obj:
        raise TypeError(f"Expected a dict payload with a 'smiles' field in {path}, got {type(obj)}")

    smiles = list(obj["smiles"])
    dataset_labels = list(map(str, obj["dataset"])) if obj.get("dataset") is not None else None
    ecfp = obj.get("ecfp")
    mol_idx = obj.get("mol_idx")
    ecfp_radius = obj.get("ecfp_radius")
    ecfp_nbits = obj.get("ecfp_nbits")

    if expected_source is not None and dataset_labels is not None:
        unique_sources = set(dataset_labels)
        if expected_source not in unique_sources:
            raise ValueError(
                f"{path} does not contain any '{expected_source}' rows "
                f"(available: {sorted(unique_sources)})."
            )
        mask = np.array([d == expected_source for d in dataset_labels])
        smiles = [s for s, m in zip(smiles, mask) if m]
        if ecfp is not None:
            ecfp = ecfp[torch.from_numpy(mask)]
        if mol_idx is not None:
            mol_idx = mol_idx[torch.from_numpy(mask)] if torch.is_tensor(mol_idx) else [v for v, m in zip(mol_idx, mask) if m]

    if not smiles:
        raise ValueError(f"No SMILES found in {path} (source={expected_source}).")

    return {
        "smiles": smiles, "ecfp": ecfp, "mol_idx": mol_idx,
        "ecfp_radius": ecfp_radius, "ecfp_nbits": ecfp_nbits,
    }


def subsample(payload: Dict[str, object], max_samples: Optional[int], rng: np.random.Generator) -> Dict[str, object]:
    n = len(payload["smiles"])
    if max_samples is None or n <= max_samples:
        return payload
    idx = rng.choice(n, size=max_samples, replace=False)
    idx_t = torch.from_numpy(idx)
    return {
        "smiles": [payload["smiles"][i] for i in idx],
        "ecfp": payload["ecfp"][idx_t] if payload["ecfp"] is not None else None,
        "mol_idx": (
            payload["mol_idx"][idx_t] if torch.is_tensor(payload["mol_idx"])
            else ([payload["mol_idx"][i] for i in idx] if payload["mol_idx"] is not None else None)
        ),
        "ecfp_radius": payload["ecfp_radius"], "ecfp_nbits": payload["ecfp_nbits"],
    }


# -----------------------------------------------------------------------------
# ECFP (only used if --force-recompute-ecfp, or the saved field is missing)
# -----------------------------------------------------------------------------


def _ecfp_worker(args) -> np.ndarray:
    smiles, radius, n_bits = args
    bits = np.zeros(n_bits, dtype=np.uint8)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return bits
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    bits[list(fp.GetOnBits())] = 1
    return bits


def compute_ecfps(smiles_list: Sequence[str], radius: int, n_bits: int, n_workers: int) -> torch.Tensor:
    tasks = [(s, radius, n_bits) for s in smiles_list]
    if n_workers <= 1:
        results = [_ecfp_worker(t) for t in tqdm(tasks, desc="ECFP (serial)")]
    else:
        with mp.Pool(n_workers) as pool:
            results = list(tqdm(pool.imap(_ecfp_worker, tasks, chunksize=256),
                                total=len(tasks), desc=f"ECFP ({n_workers} workers)"))
    return torch.from_numpy(np.stack(results, axis=0))


def resolve_ecfp(payload: Dict[str, object], radius: int, n_bits: int, n_workers: int,
                  force_recompute: bool, label: str) -> torch.Tensor:
    saved = payload["ecfp"]
    saved_radius, saved_nbits = payload["ecfp_radius"], payload["ecfp_nbits"]

    if saved is not None and not force_recompute:
        if saved_radius == radius and saved_nbits == n_bits:
            print(f"[{label}] Reusing saved ECFP (radius={saved_radius}, nBits={saved_nbits}).")
            return saved
        print(f"[{label}] Saved ECFP is radius={saved_radius}/nBits={saved_nbits}, "
              f"requested radius={radius}/nBits={n_bits} -- recomputing.")
    elif saved is None:
        print(f"[{label}] No saved ECFP field found -- computing fresh.")

    return compute_ecfps(payload["smiles"], radius, n_bits, n_workers)


# -----------------------------------------------------------------------------
# Descriptors
# -----------------------------------------------------------------------------


def _descriptor_worker(smi: str) -> Dict[str, float]:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return {name: np.nan for name in DEFAULT_DESCRIPTORS}
    return {
        "MolWt": float(Descriptors.MolWt(mol)),
        "ExactMolWt": float(Descriptors.ExactMolWt(mol)),
        "LogP": float(Descriptors.MolLogP(mol)),
        "TPSA": float(Descriptors.TPSA(mol)),
        "HeavyAtomCount": float(Descriptors.HeavyAtomCount(mol)),
        "NumAtoms": float(mol.GetNumAtoms()),
        "NumBonds": float(mol.GetNumBonds()),
        "NumHeteroatoms": float(Descriptors.NumHeteroatoms(mol)),
        "RingCount": float(Descriptors.RingCount(mol)),
        "NumAromaticRings": float(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "NumAliphaticRings": float(rdMolDescriptors.CalcNumAliphaticRings(mol)),
        "NumSaturatedRings": float(rdMolDescriptors.CalcNumSaturatedRings(mol)),
        "NumRotatableBonds": float(Descriptors.NumRotatableBonds(mol)),
        "NumHDonors": float(Descriptors.NumHDonors(mol)),
        "NumHAcceptors": float(Descriptors.NumHAcceptors(mol)),
        "FormalCharge": float(Descriptors.ForceFieldMMFF94Charge(mol) if False else Chem.GetFormalCharge(mol)),
        "FractionCSP3": float(Descriptors.FractionCSP3(mol)),
        "NumAmideBonds": float(rdMolDescriptors.CalcNumAmideBonds(mol)),
        "NumBridgeheadAtoms": float(rdMolDescriptors.CalcNumBridgeheadAtoms(mol)),
        "NumSpiroAtoms": float(rdMolDescriptors.CalcNumSpiroAtoms(mol)),
    }


def compute_descriptors(smiles_list: Sequence[str], n_workers: int) -> pd.DataFrame:
    if n_workers <= 1:
        rows = [_descriptor_worker(s) for s in tqdm(smiles_list, desc="RDKit descriptors")]
    else:
        with mp.Pool(n_workers) as pool:
            rows = list(tqdm(pool.imap(_descriptor_worker, smiles_list, chunksize=256),
                             total=len(smiles_list), desc=f"RDKit descriptors ({n_workers} workers)"))
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Plots (styled consistently with the rest of the pipeline: tab10 palette,
# alpha/size matching the correlation-scatter panels, minimal spines, no grid)
# -----------------------------------------------------------------------------


def make_pca_plot(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    sources = sorted(df["source"].unique())
    for i, source in enumerate(sources):
        sub = df[df["source"] == source]
        ax.scatter(sub["pca0"], sub["pca1"], s=2, alpha=0.1, linewidths=0,
                  color=QUALITATIVE_PALETTE[i % len(QUALITATIVE_PALETTE)],
                  label=source, rasterized=True)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("ECFP PCA: SpectraNP vs. NMRExp-full")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    leg = ax.legend(frameon=True, fontsize=9, markerscale=4)
    for lh in leg.legend_handles:
        lh.set_alpha(1.0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved {out_path}")


def make_boxplot(df: pd.DataFrame, out_path: Path) -> None:
    descs = [desc for desc in DEFAULT_DESCRIPTORS if desc in df.columns]
    n_cols = 4
    n_rows = math.ceil(len(descs) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.2 * n_rows), squeeze=False)
    sources = sorted(df["source"].unique())

    for ax, desc in zip(axes.flat, descs):
        data, labels = [], []
        for source in sources:
            vals = df.loc[df["source"] == source, desc].dropna().to_numpy()
            if len(vals) == 0:
                continue
            data.append(vals)
            labels.append(source)
        bp = ax.boxplot(data, patch_artist=True, tick_labels=labels, widths=0.6)
        for i, box in enumerate(bp["boxes"]):
            box.set(facecolor=QUALITATIVE_PALETTE[i % len(QUALITATIVE_PALETTE)], alpha=0.6)
        for median in bp["medians"]:
            median.set(color="black", linewidth=1.2)
        ax.set_title(desc, fontsize=10)
        ax.set_ylabel(desc)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes.flat[len(descs):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved {out_path}")


def make_histogram(df: pd.DataFrame, out_path: Path) -> None:
    descs = [desc for desc in DEFAULT_DESCRIPTORS if desc in df.columns]
    n_cols = 4
    n_rows = math.ceil(len(descs) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.2 * n_rows), squeeze=False)
    sources = sorted(df["source"].unique())

    for ax, desc in zip(axes.flat, descs):
        for i, source in enumerate(sources):
            vals = df.loc[df["source"] == source, desc].dropna().to_numpy()
            if len(vals) == 0:
                continue
            ax.hist(vals, bins=30, alpha=0.4, linewidth=0,
                   color=QUALITATIVE_PALETTE[i % len(QUALITATIVE_PALETTE)], label=source)
        ax.set_title(desc, fontsize=10)
        ax.set_xlabel(desc)
        ax.set_ylabel("count")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=False)

    for ax in axes.flat[len(descs):]:
        ax.axis("off")

    plt.tight_layout(rect=(0, 0.05, 1, 1))
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved {out_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    spectranp_path = args.spectranp_data or args.data_path
    nmrexp_path = args.nmrexp_data or args.data_path
    if spectranp_path is None or nmrexp_path is None:
        raise ValueError("Provide either --data-path, or both --spectranp-data "
                         "and --nmrexp-data.")

    print("=" * 78)
    print("Comparing SpectraNP vs. NMRExp chemical space (pre-extracted data only)")
    print(f"  spectranp source : {spectranp_path}"
          f"{' (filtered to dataset==spectranp)' if args.spectranp_data is None else ''}")
    print(f"  nmrexp source    : {nmrexp_path}"
          f"{' (filtered to dataset==nmrexp)' if args.nmrexp_data is None else ''}")
    print(f"  max samples/source: {args.max_samples if args.max_samples is not None else 'all'}")
    print("=" * 78)

    spectranp = subsample(
        load_payload(spectranp_path, expected_source=None if args.spectranp_data else "spectranp"),
        args.max_samples, rng,
    )
    nmrexp = subsample(
        load_payload(nmrexp_path, expected_source=None if args.nmrexp_data else "nmrexp"),
        args.max_samples, rng,
    )
    print(f"Loaded {len(spectranp['smiles'])} spectranp molecules, "
          f"{len(nmrexp['smiles'])} nmrexp molecules.")

    spectranp_ecfp = resolve_ecfp(spectranp, args.fps_radius, args.fps_bits, args.n_workers,
                                  args.force_recompute_ecfp, "spectranp")
    nmrexp_ecfp = resolve_ecfp(nmrexp, args.fps_radius, args.fps_bits, args.n_workers,
                               args.force_recompute_ecfp, "nmrexp")

    combined_smiles = spectranp["smiles"] + nmrexp["smiles"]
    combined_labels = ["spectranp"] * len(spectranp["smiles"]) + ["nmrexp"] * len(nmrexp["smiles"])

    def to_mol_idx_list(payload_mol_idx, n):
        if payload_mol_idx is None:
            return [None] * n
        return payload_mol_idx.tolist() if torch.is_tensor(payload_mol_idx) else list(payload_mol_idx)

    combined_mol_idx = (
        to_mol_idx_list(spectranp["mol_idx"], len(spectranp["smiles"]))
        + to_mol_idx_list(nmrexp["mol_idx"], len(nmrexp["smiles"]))
    )

    combined_ecfp = torch.cat([spectranp_ecfp, nmrexp_ecfp], dim=0).numpy().astype(np.float32)
    pca = PCA(n_components=2, random_state=args.seed)
    pca_coords = pca.fit_transform(combined_ecfp)
    print(f"PCA explained variance ratio: PC1={pca.explained_variance_ratio_[0]:.3f}, "
          f"PC2={pca.explained_variance_ratio_[1]:.3f}")

    meta_df = pd.DataFrame({
        "source": combined_labels, "smiles": combined_smiles, "mol_idx": combined_mol_idx,
        "pca0": pca_coords[:, 0], "pca1": pca_coords[:, 1],
    })

    descriptor_df = pd.concat([
        compute_descriptors(spectranp["smiles"], args.n_workers).assign(source="spectranp"),
        compute_descriptors(nmrexp["smiles"], args.n_workers).assign(source="nmrexp"),
    ], ignore_index=True)

    combined_df = pd.concat([meta_df, descriptor_df.loc[:, DEFAULT_DESCRIPTORS]], axis=1)

    summary_path = args.out_dir / "spectranp_vs_nmrexp_summary.csv"
    combined_df.to_csv(summary_path, index=False)
    print(f"✓ Saved {summary_path}")

    make_pca_plot(combined_df, args.out_dir / "spectranp_vs_nmrexp_pca.png")
    make_boxplot(combined_df, args.out_dir / "spectranp_vs_nmrexp_descriptor_boxplots.png")
    make_histogram(combined_df, args.out_dir / "spectranp_vs_nmrexp_descriptor_histograms.png")

    print("\nDone.")


if __name__ == "__main__":
    main()