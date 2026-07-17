#!/usr/bin/env python
"""
6a_real_v_synthetic_umap.py
==============================

Shows the real-vs-synthetic spectra embedding gap (see `real_vs_synthetic_gap.py`
for the full statistical writeup) IN CONTEXT of the whole cotrain chemical
space, in the same visual language as the original "pretty_plot" UMAP figure
(`umap_main_figure.py`'s ancestor): a big gray background scatter of every
cotrain molecule, with colored zoom-rectangle insets -- one per real-*/syn-*
benchmark pair -- each showing that pair's matched (real, synthetic-spectra)
embedding points connected by a line, plus a molecule-grid panel of the
molecules whose embedding moved the most between spectra sources.

Pipeline
--------
    1. Load the cotrain background (`<prefix>_<split>_global_cond.pt` from
       `extract_cotrain_embeddings.py`) and fit UMAP on it (or load a
       previously pickled, already-fitted reducer).
    2. For each real-*/syn-* pair (ONBOARDING.md §1.3), extract `global_cond`
       for both sides with the SAME checkpoint (reusing the model/datamodule
       loading from `real_vs_synthetic_gap.py`), align molecules by `mol_idx`
       (canonical-SMILES fallback), and project both sides into the cotrain
       UMAP space via `reducer.transform` -- NOT a fresh fit, so the pair's
       points are genuinely comparable to the background.
    3. Render:
         - one big figure: gray cotrain background + a colored rectangle,
           label, and connected real->synth point pairs for every benchmark
           pair (the "pretty_plot"-style main figure).
         - one zoom+molecule-grid inset per pair: left panel is the zoomed
           scatter (background gray, this pair's points + connecting lines),
           right panel shows the top-shifted molecules (largest real<->synth
           embedding distance) with their shift magnitude as a legend.
    4. Writes a short text summary (paired vs. background distance, gap
       ratio) per pair -- for the full property/retrieval statistical
       breakdown, use `real_vs_synthetic_gap.py` instead; this script is
       scoped to the visualization.

Usage
-----
python 6a_real_v_synthetic_umap.py --nmr3d-root /home/jc4587/3_AI4chemistr/nmr-to-3d --ckpt /projects/CRYOEM/zhonglab/data_nmr/2026/ckpts/26-05-01-cotraining-baselines/cotrain-epoch0899-accuracy60_70.ckpt --ckpt-sigma 2.8268 --cotrain-data-dir /scratch/gpfs/ZHONGE/jc4587/nmr_embs_FINAL --out-dir 6a_real_v_synthetic_umap
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

from rdkit import Chem, RDLogger
from rdkit.Chem import Draw

RDLogger.DisableLog("rdApp.*")

QUALITATIVE_PALETTE = plt.cm.tab10(np.linspace(0, 1, 10))
DPI = 600

# -----------------------------------------------------------------------------
# Default real/syn pairing (ONBOARDING.md §1.3 -- same as real_vs_synthetic_gap.py)
# -----------------------------------------------------------------------------

DEFAULT_PAIRS = [
    {"real_name": "real-fda-cdcl3", "syn_name": "syn-fda", "real_suffix": "_both", "syn_suffix": "_both"},
    {"real_name": "real-fda-dmso", "syn_name": "syn-fda", "real_suffix": "_both", "syn_suffix": "_both"},
    {"real_name": "real-5mer-d2o", "syn_name": "syn-5mer", "real_suffix": "_both", "syn_suffix": "_both"},
    {"real_name": "real-5mer-dmso", "syn_name": "syn-5mer", "real_suffix": "_both", "syn_suffix": "_both"},
    {"real_name": "real-np-mo", "syn_name": "syn-np-mo", "real_suffix": "_both", "syn_suffix": "_both"},
    {"real_name": "real-npmrd-hq", "syn_name": "syn-npmrd-hq", "real_suffix": "_both", "syn_suffix": "_both"},
    {"real_name": "real-specteach", "syn_name": "syn-specteach", "real_suffix": "_both", "syn_suffix": "_both"},
]


@dataclass
class PairSpec:
    real_name: str
    syn_name: str
    real_suffix: str
    syn_suffix: str


# -----------------------------------------------------------------------------
# Model + datamodule loading (same as real_vs_synthetic_gap.py)
# -----------------------------------------------------------------------------


def load_model(ckpt: Path, device: str):
    from src.model.model import NMRTo3DStructureElucidation

    model = NMRTo3DStructureElucidation.load_from_checkpoint(str(ckpt), map_location=device)
    model.eval()
    model.to(device)

    if hasattr(model, "model") and hasattr(model.model, "score_model"):
        score_model = model.model.score_model
    else:
        score_model = model.score_model

    return model, score_model.y_embedder


def build_datamodule(cfg_dir: str, hydra_name: str, split_suffix: str,
                      sigma_data: float, condition: str):
    from hydra import compose, initialize_config_dir
    from src.data.datamodule import NMRDataModule

    with initialize_config_dir(cfg_dir, version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                f"+data={hydra_name}",
                f"+condition={condition}",
                f"dataset_args.split_indices_suffix={split_suffix}",
                f"dataset_args.sigma_data={sigma_data}",
            ],
        )
    return NMRDataModule(cfg.dataset_args)


def get_split_dataloader(dm, split: str):
    if split in ("train", "val"):
        dm.prepare_data()
        dm.setup("fit")
        return dm.train_dataloader() if split == "train" else dm.val_dataloader()
    elif split == "test":
        dm.prepare_data()
        dm.setup("test")
        return dm.test_dataloader()
    raise ValueError(f"Unknown split: {split}")


def extract_split_embeddings(model, peak_embedder, dm, split: str, device: str) -> Dict:
    dataloader = get_split_dataloader(dm, split)

    smiles_out: List[str] = []
    mol_idx_out: List[int] = []
    global_cond_out: List[torch.Tensor] = []
    have_mol_idx = True

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"    forward pass ({split})"):
            model_inputs, smiles = batch[0]
            condition = {k: v.to(device) for k, v in model_inputs["condition"].items()}
            global_cond, _, _, _ = peak_embedder(condition, extract_all=True)

            global_cond_out.append(global_cond.cpu())
            smiles_out.extend(smiles)

            if have_mol_idx:
                mol_idx_batch = model_inputs.get("mol_idx") if isinstance(model_inputs, dict) else None
                if mol_idx_batch is not None:
                    mol_idx_out.extend(mol_idx_batch.detach().cpu().tolist())
                else:
                    have_mol_idx = False
                    mol_idx_out = []

    return {
        "smiles": smiles_out,
        "global_cond": torch.cat(global_cond_out, dim=0).numpy().astype(np.float32),
        "mol_idx": np.array(mol_idx_out, dtype=np.int64) if have_mol_idx and len(mol_idx_out) == len(smiles_out) else None,
    }


# -----------------------------------------------------------------------------
# Alignment (mol_idx primary, canonical-SMILES fallback -- same as real_vs_synthetic_gap.py)
# -----------------------------------------------------------------------------


def canonicalize(smi: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol is not None else None


def align_pair(real: Dict, syn: Dict) -> Tuple[np.ndarray, np.ndarray, str, int]:
    if real["mol_idx"] is not None and syn["mol_idx"] is not None:
        real_map = {v: i for i, v in enumerate(real["mol_idx"])}
        syn_map = {v: i for i, v in enumerate(syn["mol_idx"])}
        common = sorted(set(real_map) & set(syn_map))
        real_idx = np.array([real_map[c] for c in common])
        syn_idx = np.array([syn_map[c] for c in common])
        method = "mol_idx"
    else:
        real_canon = [canonicalize(s) for s in real["smiles"]]
        syn_canon = [canonicalize(s) for s in syn["smiles"]]
        syn_map = {}
        for i, c in enumerate(syn_canon):
            if c is not None and c not in syn_map:
                syn_map[c] = i
        real_idx_list, syn_idx_list = [], []
        for i, c in enumerate(real_canon):
            if c is not None and c in syn_map:
                real_idx_list.append(i)
                syn_idx_list.append(syn_map[c])
        real_idx = np.array(real_idx_list)
        syn_idx = np.array(syn_idx_list)
        method = "canonical_smiles"

    n_mismatch = 0
    for ri, si in zip(real_idx, syn_idx):
        if canonicalize(real["smiles"][ri]) != canonicalize(syn["smiles"][si]):
            n_mismatch += 1

    return real_idx, syn_idx, method, n_mismatch


# -----------------------------------------------------------------------------
# Cotrain background loading
# -----------------------------------------------------------------------------


def load_cotrain_background(data_dir: Path, prefix: str, splits: Sequence[str],
                             embedding_key: str) -> np.ndarray:
    embed_parts: List[torch.Tensor] = []
    for split in splits:
        path = data_dir / f"{prefix}_{split}_global_cond.pt"
        if not path.exists():
            raise FileNotFoundError(f"Expected extraction output not found: {path}")
        print(f"Loading cotrain background: {path} ...")
        d = torch.load(path, map_location="cpu")
        if embedding_key not in d:
            raise KeyError(f"'{embedding_key}' not found in {path}. Keys: {list(d.keys())}")
        embed_parts.append(d[embedding_key])
    return torch.cat(embed_parts, dim=0).numpy().astype(np.float32)


# -----------------------------------------------------------------------------
# UMAP: fit (or load) on the cotrain background; keep a single picklable
# "projector" object (scaler -> optional PCA -> UMAP) so any new points
# (the real/synth pairs) go through EXACTLY the same preprocessing chain the
# background was fit through. Calling `.transform(raw_768d_embedding)` on
# this object does the whole chain -- callers never touch the scaler/PCA
# directly, which is what caused the original bug (a bare dict of
# {scaler, pca, umap} has no `.transform`, and skipping scaler/PCA for new
# points would project them into the wrong space entirely).
# -----------------------------------------------------------------------------


class FittedProjector:
    """Bundles the fitted preprocessing chain (StandardScaler -> optional
    PCA -> UMAP) behind a single `.transform`, so background and pair
    embeddings are always projected identically."""

    def __init__(self, scaler, pca, umap_reducer):
        self.scaler = scaler
        self.pca = pca
        self.umap_reducer = umap_reducer

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.scaler is not None:
            x = self.scaler.transform(x)
        if self.pca is not None:
            x = self.pca.transform(x)
        return self.umap_reducer.transform(x)


def fit_or_load_umap(embedding: np.ndarray, n_neighbors: int, min_dist: float,
                      metric: str, seed: int, save_model_path: Optional[Path],
                      load_model_path: Optional[Path], pca_components: Optional[int],
                      use_scaler: bool) -> Tuple["FittedProjector", np.ndarray]:
    if load_model_path is not None:
        print(f"Loading fitted projector (scaler/PCA/UMAP) from {load_model_path}")
        with open(load_model_path, "rb") as f:
            projector = pickle.load(f)
        if not isinstance(projector, FittedProjector):
            raise TypeError(
                f"{load_model_path} does not contain a FittedProjector "
                f"(got {type(projector)}) -- if this was pickled by an "
                f"older/hand-edited version of this script, re-fit and "
                f"re-save it with the current code."
            )
        # Reuse the UMAP object's own fit-time embedding rather than calling
        # .transform() on the training data again (transform is an
        # approximate re-projection via nearest neighbors and can differ
        # slightly from the exact fit-time layout).
        return projector, projector.umap_reducer.embedding_

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    import umap.umap_ as umap_lib

    x = embedding
    scaler = None
    if use_scaler:
        print(f"Fitting StandardScaler on {x.shape[0]} points ({x.shape[1]} dims)...")
        scaler = StandardScaler()
        x = scaler.fit_transform(x)

    pca = None
    if pca_components:
        print(f"Fitting PCA ({pca_components} components)...")
        pca = PCA(n_components=pca_components, random_state=seed)
        x = pca.fit_transform(x)
        print(f"  PCA({pca_components}) explains {pca.explained_variance_ratio_.sum() * 100:.1f}% of variance")

    print(f"Fitting UMAP on {'PCA-reduced' if pca is not None else 'scaled' if scaler is not None else 'raw'} "
          f"background ({x.shape[0]} points, n_neighbors={n_neighbors}, "
          f"min_dist={min_dist}, metric={metric}) ...")
    umap_reducer = umap_lib.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                                  metric=metric, random_state=seed)
    emb2d = umap_reducer.fit_transform(x)

    projector = FittedProjector(scaler, pca, umap_reducer)

    if save_model_path is not None:
        save_model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_model_path, "wb") as f:
            pickle.dump(projector, f)
        print(f"✓ Cached fitted projector to {save_model_path}")

    return projector, emb2d


# -----------------------------------------------------------------------------
# Per-pair result: aligned 2D coords + high-dim shift statistics
# -----------------------------------------------------------------------------


@dataclass
class PairResult:
    label: str
    n_matched: int
    n_smiles_mismatch: int
    align_method: str
    real_2d: np.ndarray
    syn_2d: np.ndarray
    matched_smiles: List[str]
    paired_euclidean: np.ndarray       # high-dim (not 2D) real<->syn distance per molecule
    background_euclidean: np.ndarray   # high-dim real_i vs. mismatched syn_j
    color: np.ndarray
    center: Tuple[float, float]
    half_extent: float


def compute_pair_result(label: str, real: Dict, syn: Dict, reducer,
                         color: np.ndarray, window_size: Optional[float],
                         min_window: float, padding_frac: float) -> Optional[PairResult]:
    real_idx, syn_idx, method, n_mismatch = align_pair(real, syn)
    n = len(real_idx)
    if n < 1:
        print(f"[warn] {label}: 0 matched molecules (alignment={method}), skipping.")
        return None
    if n_mismatch > 0:
        print(f"[WARNING] {label}: {n_mismatch}/{n} matched pairs have DIFFERENT "
              f"canonical SMILES -- alignment may be wrong!")

    real_emb = real["global_cond"][real_idx]
    syn_emb = syn["global_cond"][syn_idx]
    matched_smiles = [real["smiles"][i] for i in real_idx]

    real_2d = reducer.transform(real_emb)
    syn_2d = reducer.transform(syn_emb)

    paired_euclidean = np.linalg.norm(real_emb - syn_emb, axis=1)

    if n >= 2:
        euclid_full = np.linalg.norm(real_emb[:, None, :] - syn_emb[None, :, :], axis=2)
        off_mask = ~np.eye(n, dtype=bool)
        background_euclidean = euclid_full[off_mask]
    else:
        background_euclidean = np.array([])

    combined_2d = np.concatenate([real_2d, syn_2d], axis=0)
    if window_size is not None:
        center = tuple(combined_2d.mean(axis=0))
        half_extent = window_size / 2
    else:
        center_arr = combined_2d.mean(axis=0)
        max_extent = float(np.abs(combined_2d - center_arr).max()) if len(combined_2d) else min_window / 2
        half_extent = max(max_extent * (1.0 + padding_frac), min_window / 2)
        center = tuple(center_arr)

    return PairResult(
        label=label, n_matched=n, n_smiles_mismatch=n_mismatch, align_method=method,
        real_2d=real_2d, syn_2d=syn_2d, matched_smiles=matched_smiles,
        paired_euclidean=paired_euclidean, background_euclidean=background_euclidean,
        color=color, center=center, half_extent=half_extent,
    )


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def plot_main_scatter(cotrain_emb2d: np.ndarray, pair_results: List[PairResult],
                       out_path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.scatter(cotrain_emb2d[:, 0], cotrain_emb2d[:, 1], color="lightgray",
              s=0.25, alpha=0.25, linewidths=0, zorder=1)

    for pr in pair_results:
        for i in range(pr.n_matched):
            ax.plot([pr.real_2d[i, 0], pr.syn_2d[i, 0]], [pr.real_2d[i, 1], pr.syn_2d[i, 1]],
                   color=pr.color, alpha=0.5, linewidth=0.8, zorder=2)

        ax.scatter(pr.real_2d[:, 0], pr.real_2d[:, 1], marker="o", facecolor=pr.color,
                  edgecolor="black", linewidths=0.3, s=28, zorder=3)
        ax.scatter(pr.syn_2d[:, 0], pr.syn_2d[:, 1], marker="^", facecolor=pr.color,
                  edgecolor="black", linewidths=0.3, s=28, zorder=3)

        rect = Rectangle(
            (pr.center[0] - pr.half_extent, pr.center[1] - pr.half_extent),
            2 * pr.half_extent, 2 * pr.half_extent,
            fill=False, linewidth=2, edgecolor=pr.color, zorder=4,
        )
        ax.add_patch(rect)
        ax.text(pr.center[0], pr.center[1] + pr.half_extent + 0.3, pr.label,
               color=pr.color, ha="center", fontsize=9, weight="bold", zorder=5)

    shape_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                  markeredgecolor="black", markersize=8, label="real"),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="gray",
                  markeredgecolor="black", markersize=8, label="synthetic"),
    ]
    ax.legend(handles=shape_handles, loc="best", frameon=True, fontsize=9)

    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title("Real vs. synthetic spectra embedding shift, in cotrain UMAP context")
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved {out_path}")


def mol_grid_for_top_shifted(pr: PairResult, n_mols: int):
    order = np.argsort(-pr.paired_euclidean)[:n_mols]
    mols, legends = [], []
    for idx in order:
        mol = Chem.MolFromSmiles(pr.matched_smiles[idx])
        if mol is None:
            continue
        mols.append(mol)
        legends.append(f"\u0394={pr.paired_euclidean[idx]:.2f}")
    if not mols:
        return None
    return Draw.MolsToGridImage(mols, molsPerRow=min(3, len(mols)), subImgSize=(220, 220),
                                legends=legends, returnPNG=False)


def plot_pair_inset(pr: PairResult, cotrain_emb2d: np.ndarray, n_mols: int,
                     out_dir: Path, dpi: int) -> None:
    fig = plt.figure(figsize=(9, 4.5))
    gs = GridSpec(1, 2, width_ratios=[1, 1], figure=fig)

    ax_scatter = fig.add_subplot(gs[0, 0])
    ax_scatter.scatter(cotrain_emb2d[:, 0], cotrain_emb2d[:, 1], color="lightgray",
                       s=1, alpha=0.5, linewidths=0, zorder=1)

    for i in range(pr.n_matched):
        ax_scatter.plot([pr.real_2d[i, 0], pr.syn_2d[i, 0]], [pr.real_2d[i, 1], pr.syn_2d[i, 1]],
                       color=pr.color, alpha=0.6, linewidth=1.0, zorder=2)
    ax_scatter.scatter(pr.real_2d[:, 0], pr.real_2d[:, 1], marker="o", facecolor=pr.color,
                       edgecolor="black", linewidths=0.4, s=45, zorder=3, label="real")
    ax_scatter.scatter(pr.syn_2d[:, 0], pr.syn_2d[:, 1], marker="^", facecolor=pr.color,
                       edgecolor="black", linewidths=0.4, s=45, zorder=3, label="synthetic")

    ax_scatter.set_xlim(pr.center[0] - pr.half_extent, pr.center[0] + pr.half_extent)
    ax_scatter.set_ylim(pr.center[1] - pr.half_extent, pr.center[1] + pr.half_extent)
    ax_scatter.set_aspect("equal")
    ax_scatter.set_xticks([])
    ax_scatter.set_yticks([])
    ax_scatter.set_title(f"{pr.label} (n={pr.n_matched})")
    ax_scatter.legend(loc="best", frameon=True, fontsize=8)

    mol_img = mol_grid_for_top_shifted(pr, n_mols)
    ax_mols = fig.add_subplot(gs[0, 1])
    if mol_img is not None:
        ax_mols.imshow(mol_img)
    else:
        ax_mols.text(0.5, 0.5, "no valid SMILES", ha="center", va="center", transform=ax_mols.transAxes)
    ax_mols.set_xticks([])
    ax_mols.set_yticks([])
    ax_mols.set_title("most-shifted molecules (real\u2192synthetic)")

    plt.tight_layout()
    out_path = out_dir / f"inset_{pr.label}.png"
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    if mol_img is not None:
        mol_img.save(out_dir / f"inset_{pr.label}_mols.png")

    print(f"✓ Saved {out_path}")


def write_summary(pair_results: List[PairResult], out_path: Path) -> None:
    lines = [
        "Real vs. synthetic spectra: UMAP-context shift summary",
        "=" * 72,
        "",
        "gap ratio = mean(paired real<->syn distance) / mean(background "
        "mismatched-pair distance), both computed in the ORIGINAL high-dim "
        "embedding space (not the 2D UMAP projection, which can distort "
        "distances). See real_vs_synthetic_gap.py for the full statistical "
        "breakdown (retrieval accuracy, property correlations, etc).",
        "",
    ]
    for pr in pair_results:
        gap_ratio = (
            float(pr.paired_euclidean.mean() / (pr.background_euclidean.mean() + 1e-12))
            if len(pr.background_euclidean) else float("nan")
        )
        lines.append(f"--- {pr.label} (n={pr.n_matched}, alignment={pr.align_method}) ---")
        if pr.n_smiles_mismatch > 0:
            lines.append(f"  !! {pr.n_smiles_mismatch} SMILES mismatches in matched pairs.")
        lines.append(f"  mean paired distance     : {pr.paired_euclidean.mean():.4f}")
        if len(pr.background_euclidean):
            lines.append(f"  mean background distance : {pr.background_euclidean.mean():.4f}")
            lines.append(f"  gap ratio                : {gap_ratio:.3f}")
        lines.append("")

    out_path.write_text("\n".join(lines))
    print(f"✓ Saved {out_path}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="UMAP visualization of the real-vs-synthetic spectra "
                     "embedding gap in the context of the full cotrain "
                     "chemical space.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--nmr3d-root", type=Path, required=True)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--ckpt-sigma", type=float, required=True)
    p.add_argument("--condition", type=str, default="hcpeak")
    p.add_argument("--real-syn-split", type=str, default="test", choices=["train", "val", "test"],
                   help="Split to extract within each real-*/syn-* benchmark dataset.")
    p.add_argument("--pairs-json", type=Path, default=None)

    p.add_argument("--cotrain-data-dir", type=Path, required=True,
                   help="Directory with <prefix>_<split>_global_cond.pt "
                        "(extract_cotrain_embeddings.py output) used as the "
                        "gray background.")
    p.add_argument("--cotrain-prefix", type=str, default="cotrain")
    p.add_argument("--cotrain-splits", nargs="+", default=["train"],
                   choices=["train", "val", "test"])
    p.add_argument("--embedding-key", type=str, default="global_cond")

    p.add_argument("--umap-n-neighbors", type=int, default=30)
    p.add_argument("--umap-min-dist", type=float, default=0.1)
    p.add_argument("--umap-metric", type=str, default="cosine")
    p.add_argument("--pca-components", type=int, default=50,
                   help="Pre-reduce the (scaled) background embedding to "
                        "this many PCA components before fitting UMAP -- "
                        "makes UMAP tractable at cotrain scale (millions of "
                        "points). Pass 0 to skip PCA and run UMAP directly "
                        "on the (scaled) embedding.")
    p.add_argument("--no-scale", action="store_true",
                   help="Skip StandardScaler before PCA/UMAP (on by default).")
    p.add_argument("--save-umap-model", type=Path, default=None,
                   help="Pickle the fitted projector (scaler/PCA/UMAP) here "
                        "for reuse.")
    p.add_argument("--load-umap-model", type=Path, default=None,
                   help="Load a previously pickled fitted-on-cotrain "
                        "projector instead of fitting fresh.")

    p.add_argument("--window-size", type=float, default=None,
                   help="Fixed zoom-window side length for every pair's "
                        "inset. Default: auto-fit per pair (bounding extent "
                        "of its points + padding).")
    p.add_argument("--min-window-size", type=float, default=1.5,
                   help="Floor on the auto-fit window size, so pairs with "
                        "very few/tight points still get a visible inset.")
    p.add_argument("--window-padding-frac", type=float, default=0.35)
    p.add_argument("--n-mols-per-inset", type=int, default=6)

    p.add_argument("--out-dir", type=Path, default=Path("6a_real_v_synthetic_umap"))
    p.add_argument("--dpi", type=int, default=DPI)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(args.nmr3d_root))
    config_dir = str(args.nmr3d_root / "configs")

    pairs_raw = json.loads(Path(args.pairs_json).read_text()) if args.pairs_json else DEFAULT_PAIRS
    pairs = [PairSpec(**p) for p in pairs_raw]

    print("=" * 78)
    print("Real vs. synthetic spectra: UMAP-context visualization")
    print(f"  ckpt   : {args.ckpt}")
    print(f"  sigma  : {args.ckpt_sigma}")
    print(f"  pairs  : {[(p.real_name, p.syn_name) for p in pairs]}")
    print("=" * 78)

    cotrain_embedding = load_cotrain_background(
        args.cotrain_data_dir, args.cotrain_prefix, args.cotrain_splits, args.embedding_key,
    )
    reducer, cotrain_emb2d = fit_or_load_umap(
        cotrain_embedding, args.umap_n_neighbors, args.umap_min_dist, args.umap_metric,
        args.seed, args.save_umap_model, args.load_umap_model,
        pca_components=(args.pca_components or None), use_scaler=(not args.no_scale),
    )

    model, peak_embedder = load_model(args.ckpt, args.device)

    pair_results: List[PairResult] = []
    for i, pair in enumerate(pairs):
        label = f"{pair.real_name}_vs_{pair.syn_name}"
        print(f"\n### {label} ###")
        color = QUALITATIVE_PALETTE[i % len(QUALITATIVE_PALETTE)]

        try:
            real_dm = build_datamodule(config_dir, pair.real_name, pair.real_suffix, args.ckpt_sigma, args.condition)
            real_data = extract_split_embeddings(model, peak_embedder, real_dm, args.real_syn_split, args.device)

            syn_dm = build_datamodule(config_dir, pair.syn_name, pair.syn_suffix, args.ckpt_sigma, args.condition)
            syn_data = extract_split_embeddings(model, peak_embedder, syn_dm, args.real_syn_split, args.device)
        except Exception as e:
            print(f"[warn] Failed to load/extract {label}: {e}. Skipping this pair.")
            continue

        result = compute_pair_result(
            label, real_data, syn_data, reducer, color,
            args.window_size, args.min_window_size, args.window_padding_frac,
        )
        if result is not None:
            pair_results.append(result)

    if not pair_results:
        print("\n[warn] No pairs produced usable results -- nothing to plot.")
        return

    plot_main_scatter(cotrain_emb2d, pair_results, args.out_dir / "umap_real_vs_synthetic_main.png", args.dpi)

    for pr in pair_results:
        plot_pair_inset(pr, cotrain_emb2d, args.n_mols_per_inset, args.out_dir, args.dpi)

    write_summary(pair_results, args.out_dir / "real_vs_synthetic_umap_summary.txt")

    print("\nDone.")


if __name__ == "__main__":
    main()