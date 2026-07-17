#!/usr/bin/env python
"""
pca_traversal_and_correlations.py
====================================

Generalizes the ad-hoc "SLIDE 3 PC TRAVERSALS" / "SLIDE 4-5 PC CORRELATIONS"
sections of the earlier group-meeting analysis script into a single,
reproducible tool that:

    1. Fits PCA (top N components, N chosen by the user) on the embeddings
       extracted by `extract_cotrain_embeddings.py`
       (`<prefix>_<split>_global_cond.pt`).
    2. For each of the top N PCs, does a "traversal": step that PC across
       its observed range while holding every other PC fixed at its median,
       find the nearest real molecule to each step, and render a
       molecule-image-strip + PC-space-position figure
       (`pc{i}_traversal.png`).
    3. Computes every PC's Pearson correlation against a fixed panel of
       RDKit descriptors, plots each PC against its single
       best-correlated descriptor (`pca_correlations.png`), and dumps the
       *complete* PC-x-descriptor correlation matrix (not just the best
       match) to both a machine-readable CSV and a human-readable text
       table (`pc_descriptor_correlations.csv` / `.txt`).

Unlike the original script, nothing here is hardcoded to a specific
checkpoint's saved `.pt` files or a fixed PC count -- pass `--n-pcs`,
`--embedding-key`, `--data-dir`, etc.

Usage
-----
python 2_trav_and_correlations.py --data-dir /scratch/gpfs/ZHONGE/jc4587/nmr_embs_FINAL --prefix cotrain --splits train --n-pcs 8 --out-dir pca_summary
"""

from __future__ import annotations

import argparse
import io
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.ML.Descriptors import MoleculeDescriptors

# -----------------------------------------------------------------------------
# Fixed descriptor panel (same set as the earlier analysis script, so results
# stay comparable across model/dataset iterations).
# -----------------------------------------------------------------------------

DEFAULT_DESCRIPTOR_NAMES = [
    "MaxAbsEStateIndex", "MaxEStateIndex", "MinAbsEStateIndex", "MinEStateIndex",
    "qed", "SPS", "MolWt", "HeavyAtomMolWt", "ExactMolWt", "NumValenceElectrons",
    "NumRadicalElectrons", "BalabanJ", "BertzCT", "HallKierAlpha", "LabuteASA",
    "TPSA", "FractionCSP3", "HeavyAtomCount", "NHOHCount", "NOCount",
    "NumAliphaticCarbocycles", "NumAliphaticHeterocycles", "NumAliphaticRings",
    "NumAromaticCarbocycles", "NumAromaticHeterocycles", "NumAromaticRings",
    "NumHAcceptors", "NumHDonors", "NumHeteroatoms", "NumRotatableBonds",
    "NumSaturatedCarbocycles", "NumSaturatedHeterocycles", "NumSaturatedRings",
    "RingCount", "MolLogP",
]


# -----------------------------------------------------------------------------
# Data loading (same convention as the other scripts in this pipeline)
# -----------------------------------------------------------------------------


def load_cotrain_data(data_dir: Path, prefix: str, splits: Sequence[str],
                       embedding_key: str) -> Dict:
    smiles: List[str] = []
    dataset: List[str] = []
    embed_parts: List[torch.Tensor] = []

    for split in splits:
        path = data_dir / f"{prefix}_{split}_global_cond.pt"
        if not path.exists():
            raise FileNotFoundError(f"Expected extraction output not found: {path}")
        print(f"Loading {path} ...")
        d = torch.load(path, map_location="cpu")

        if embedding_key not in d:
            raise KeyError(
                f"'{embedding_key}' not found in {path}. Available keys: "
                f"{list(d.keys())}. (Was --save-layer-reps used at extraction "
                f"time, if you asked for 'layer_reps'?)"
            )

        smiles.extend(d["smiles"])
        dataset.extend(d["dataset"])
        embed_parts.append(d[embedding_key])

    embedding = torch.cat(embed_parts, dim=0)
    if embedding.dim() != 2:
        raise ValueError(
            f"Expected a 2D [N, D] tensor for '{embedding_key}', got shape "
            f"{tuple(embedding.shape)}. Pooled embeddings only (e.g. "
            f"'global_cond'); reduce/select a layer first for anything else."
        )

    return {
        "smiles": smiles,
        "dataset": dataset,
        "embedding": embedding.numpy().astype(np.float32),
    }


# -----------------------------------------------------------------------------
# PCA
# -----------------------------------------------------------------------------


def fit_pca(embedding: np.ndarray, n_components: int, scale: bool
            ) -> Tuple[np.ndarray, PCA, Optional[StandardScaler]]:
    scaler = None
    x = embedding
    if scale:
        scaler = StandardScaler()
        x = scaler.fit_transform(x)

    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(x)
    print("Explained variance ratio per PC: "
          + ", ".join(f"PC{i+1}={v:.3f}" for i, v in enumerate(pca.explained_variance_ratio_)))
    return pcs, pca, scaler


# -----------------------------------------------------------------------------
# RDKit descriptors (parallel, via a per-worker cached calculator)
# -----------------------------------------------------------------------------

_CALC = None  # set once per worker process


def _init_descriptor_worker(descriptor_names: List[str]) -> None:
    global _CALC
    _CALC = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)


def _descriptor_worker(smi: str) -> List[float]:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return [np.nan] * len(_CALC.descriptorNames)
    vals = _CALC.CalcDescriptors(mol)
    return [np.nan if (v is None or not np.isfinite(v)) else float(v) for v in vals]


def compute_descriptor_matrix(smiles_list: List[str], descriptor_names: List[str],
                               n_workers: int) -> np.ndarray:
    if n_workers <= 1:
        _init_descriptor_worker(descriptor_names)
        rows = [_descriptor_worker(s) for s in tqdm(smiles_list, desc="RDKit descriptors")]
    else:
        with mp.Pool(n_workers, initializer=_init_descriptor_worker,
                      initargs=(descriptor_names,)) as pool:
            rows = list(
                tqdm(pool.imap(_descriptor_worker, smiles_list, chunksize=256),
                     total=len(smiles_list),
                     desc=f"RDKit descriptors ({n_workers} workers)")
            )
    return np.array(rows, dtype=np.float64)


# -----------------------------------------------------------------------------
# PC <-> descriptor correlations (full matrix, plus per-PC best match)
# -----------------------------------------------------------------------------


def compute_pc_descriptor_correlations(pcs: np.ndarray, descriptor_matrix: np.ndarray,
                                        descriptor_names: List[str]) -> pd.DataFrame:
    n_pcs = pcs.shape[1]
    corr = np.full((n_pcs, len(descriptor_names)), np.nan)

    for pc_idx in tqdm(range(n_pcs), desc="Computing PC-descriptor correlations"):
        pc_values = pcs[:, pc_idx]
        for d_idx, _ in enumerate(descriptor_names):
            desc_values = descriptor_matrix[:, d_idx]
            mask = np.isfinite(desc_values) & np.isfinite(pc_values)
            if mask.sum() < 10:
                continue
            r, _ = pearsonr(pc_values[mask], desc_values[mask])
            corr[pc_idx, d_idx] = r

    return pd.DataFrame(corr, index=[f"PC{i+1}" for i in range(n_pcs)],
                         columns=descriptor_names)


def best_descriptor_per_pc(corr_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pc in corr_df.index:
        row = corr_df.loc[pc]
        abs_row = row.abs()
        if abs_row.isna().all():
            rows.append({"pc": pc, "descriptor": None, "r": np.nan})
            continue
        best_desc = abs_row.idxmax()
        rows.append({"pc": pc, "descriptor": best_desc, "r": row[best_desc]})
    return pd.DataFrame(rows)


def save_correlation_outputs(corr_df: pd.DataFrame, out_dir: Path, prefix: str) -> None:
    csv_path = out_dir / f"{prefix}_pc_descriptor_correlations.csv"
    txt_path = out_dir / f"{prefix}_pc_descriptor_correlations.txt"

    corr_df.to_csv(csv_path)

    with open(txt_path, "w") as f:
        f.write("Raw Pearson correlation (r) between each PC and each RDKit "
                "descriptor.\nRows = principal components, columns = "
                "descriptors.\n\n")
        f.write(corr_df.round(4).to_string())
        f.write("\n")

    print(f"✓ Saved {csv_path}")
    print(f"✓ Saved {txt_path}")


def plot_best_correlations(pcs: np.ndarray, descriptor_matrix: np.ndarray,
                            descriptor_names: List[str], best_df: pd.DataFrame,
                            out_path: Path) -> None:
    n_pcs = len(best_df)
    ncols = min(4, n_pcs)
    nrows = int(np.ceil(n_pcs / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(2.4 * ncols, 2.6 * nrows),
                              constrained_layout=True, squeeze=False)

    for i in range(nrows * ncols):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        if i >= n_pcs:
            ax.axis("off")
            continue

        row = best_df.iloc[i]
        if row["descriptor"] is None:
            ax.axis("off")
            continue

        d_idx = descriptor_names.index(row["descriptor"])
        desc_values = descriptor_matrix[:, d_idx]
        pc_values = pcs[:, i]
        mask = np.isfinite(desc_values) & np.isfinite(pc_values)

        ax.scatter(pc_values[mask], desc_values[mask], alpha=0.1, s=2,
                   linewidths=0, rasterized=True)
        ax.set_title(f"{row['descriptor']}\nr={row['r']:.3f}", fontsize=9)
        ax.set_xlabel(f"PC{i+1}")
        ax.set_ylabel(row["descriptor"])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved {out_path}")


# -----------------------------------------------------------------------------
# PC traversal (generalized to N PCs; others held at median)
# -----------------------------------------------------------------------------


def mol_to_image(smiles: str, size: int = 300) -> Optional[np.ndarray]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    drawer = rdMolDraw2D.MolDraw2DCairo(size, size)
    opts = drawer.drawOptions()
    opts.bondLineWidth = 2.5
    opts.padding = 0.12
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    img = Image.open(io.BytesIO(drawer.GetDrawingText()))
    return np.array(img)


def nearest_molecule(query_point: np.ndarray, pcs: np.ndarray,
                      smiles_list: List[str]) -> Tuple[str, int]:
    dists = np.linalg.norm(pcs - query_point, axis=1)
    idx = int(np.argmin(dists))
    return smiles_list[idx], idx


def pc_traversal_plot(pc_idx: int, pcs: np.ndarray, smiles: List[str],
                       n_steps: int, out_path: Path, mol_size: int = 260) -> None:
    """Step PC `pc_idx` across its range, holding every other (of the fitted)
    PCs fixed at its median; nearest-molecule + PC-space position per step."""
    n_pcs = pcs.shape[1]
    medians = np.median(pcs, axis=0)

    pc_vals = np.linspace(pcs[:, pc_idx].min(), pcs[:, pc_idx].max(), n_steps)

    fig, axes = plt.subplots(2, n_steps, figsize=(n_steps * 2, 4))
    if n_steps == 1:
        axes = axes.reshape(2, 1)

    for i, val in enumerate(pc_vals):
        query = medians.copy()
        query[pc_idx] = val

        smi, _ = nearest_molecule(query, pcs, smiles)
        img = mol_to_image(smi, size=mol_size)

        if img is not None:
            axes[0, i].imshow(img)
        axes[0, i].axis("off")
        axes[0, i].set_title(f"PC{pc_idx+1}={val:.2f}", fontsize=6)

        # Position plotted against the next PC if it exists, else PC1, so a
        # single 2D scatter can still show where the query sits even when
        # n_pcs > 2 (the traversal itself varies only pc_idx; all displayed
        # coordinates besides pc_idx are just for visual context).
        other_dim = (pc_idx + 1) % n_pcs if n_pcs > 1 else pc_idx
        axes[1, i].scatter(pcs[:, pc_idx], pcs[:, other_dim], s=0.3, alpha=0.2, c="lightgray")
        axes[1, i].scatter(query[pc_idx], query[other_dim], c="red", s=30, zorder=5)
        axes[1, i].set_xlabel(f"PC{pc_idx+1}", fontsize=6)
        axes[1, i].set_ylabel(f"PC{other_dim+1}", fontsize=6)
        axes[1, i].tick_params(labelsize=5)
        for sp in ["top", "right"]:
            axes[1, i].spines[sp].set_visible(False)

    fig.suptitle(f"PC{pc_idx+1} traversal (all other PCs fixed at median)", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved {out_path}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PCA traversal + PC-descriptor correlation analysis on "
                     "cotrain embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", type=Path, required=True,
                   help="Directory with <prefix>_<split>_global_cond.pt "
                        "(output of extract_cotrain_embeddings.py).")
    p.add_argument("--prefix", type=str, default="cotrain")
    p.add_argument("--splits", nargs="+", default=["train"],
                   choices=["train", "val", "test"])
    p.add_argument("--embedding-key", type=str, default="global_cond",
                   help="Which saved tensor to run PCA on (must be a 2D "
                        "[N, D] tensor in the .pt file), e.g. 'global_cond' "
                        "or 'ecfp'.")
    p.add_argument("--n-pcs", type=int, default=8,
                   help="Number of top principal components to analyze "
                        "(drives both the traversal and correlation steps).")
    p.add_argument("--no-scale", action="store_true",
                   help="Skip StandardScaler before PCA (on by default).")
    p.add_argument("--n-steps", type=int, default=8,
                   help="Number of steps per PC traversal.")
    p.add_argument("--out-dir", type=Path, default=Path("pca_summary"))
    p.add_argument("--descriptor-names", nargs="+", default=None,
                   help="Override the default RDKit descriptor panel used "
                        "for correlations.")
    p.add_argument("--n-desc-workers", type=int, default=max(1, mp.cpu_count() - 2))
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    descriptor_names = args.descriptor_names or DEFAULT_DESCRIPTOR_NAMES

    print("=" * 78)
    print("PCA traversal + descriptor correlation analysis")
    print(f"  data dir       : {args.data_dir}")
    print(f"  embedding key  : {args.embedding_key}")
    print(f"  n_pcs          : {args.n_pcs}")
    print(f"  scale          : {not args.no_scale}")
    print("=" * 78)

    data = load_cotrain_data(args.data_dir, args.prefix, args.splits, args.embedding_key)
    smiles = data["smiles"]
    embedding = data["embedding"]
    print(f"Loaded {len(smiles)} molecules, embedding shape={embedding.shape}")

    pcs, pca, _ = fit_pca(embedding, args.n_pcs, scale=not args.no_scale)

    # ---- Descriptors + correlations ----
    descriptor_matrix = compute_descriptor_matrix(smiles, descriptor_names, args.n_desc_workers)

    corr_df = compute_pc_descriptor_correlations(pcs, descriptor_matrix, descriptor_names)
    save_correlation_outputs(corr_df, args.out_dir, args.prefix)

    best_df = best_descriptor_per_pc(corr_df)
    best_df.to_csv(args.out_dir / f"{args.prefix}_pc_best_descriptor.csv", index=False)
    print(f"✓ Saved {args.out_dir / f'{args.prefix}_pc_best_descriptor.csv'}")

    plot_best_correlations(
        pcs, descriptor_matrix, descriptor_names, best_df,
        args.out_dir / f"{args.prefix}_pca_correlations.png",
    )

    # ---- Traversals ----
    for pc_idx in range(args.n_pcs):
        pc_traversal_plot(
            pc_idx, pcs, smiles, args.n_steps,
            args.out_dir / f"{args.prefix}_pc{pc_idx+1}_traversal.png",
        )

    print("Done.")


if __name__ == "__main__":
    main()
