#!/usr/bin/env python
"""
real_vs_synthetic_gap.py
===========================

Quantifies the "real vs. synthetic spectra gap": for each real-* / syn-*
benchmark pair (ONBOARDING.md §1.3's "Synthetic counterparts (paired
ablations)"), the *same molecules* appear in both, with real (experimental)
spectra in one and synthetic (predicted) spectra in the other. Any
embedding difference between a molecule's real-spectra run and its
synthetic-spectra run is therefore attributable to the spectra source
alone, not to structural distribution shift -- unlike the train/val/test
analysis, which conflates both.

For each pair, this script:
    1. Extracts `global_cond` embeddings for both the real-* and syn-*
       dataset (same checkpoint), aligning molecules by `mol_idx` (global
       across all nmr-to-3d datasets per ONBOARDING.md §1.1), falling back
       to canonical-SMILES matching if `mol_idx` isn't exposed by the
       dataloader.
    2. Sanity-checks the alignment: matched (real, syn) SMILES should be
       identical (same molecule) -- any mismatch is flagged as a data
       integrity problem, not silently ignored.
    3. Computes, per pair (exact, not sampled -- these benchmark sets are
       small):
         - paired embedding distance/similarity (real_i vs. its own syn_i)
         - a "background" distance/similarity distribution (real_i vs. all
           *other* syn_j, j != i) as the natural null: "how far apart are
           two different molecules, typically"
         - a gap ratio = paired / background, the same style of
           normalized shift metric used in the train/val/test analysis
         - real->syn and syn->real top-1/top-5 nearest-neighbor identity
           retrieval accuracy (does the model's embedding still recognize
           "this is the same molecule" across spectra source?)
         - correlation between paired-distance and molecular properties
           (does the gap grow with molecular complexity?)
    4. Renders, per pair and pooled overall: a PCA drift plot (real/syn
       points joined by a line per molecule), a paired-vs-background
       distance histogram, a property-correlation scatter grid, and a
       retrieval-accuracy bar chart -- in the same visual conventions
       (dpi=600, tight_layout, minimal spines, tab10 qualitative palette,
       alpha/size choices matching the correlation-scatter panels) used
       throughout this pipeline.
    5. Writes one comprehensive text report covering every pair plus
       pooled statistics.

Usage
-----
python 6_real_v_synthetic.py --nmr3d-root /home/jc4587/3_AI4chemistr/nmr-to-3d --ckpt /projects/CRYOEM/zhonglab/data_nmr/2026/ckpts/26-05-01-cotraining-baselines/cotrain-epoch0899-accuracy60_70.ckpt --ckpt-sigma 2.8268 --out-dir 6_real_vs_synthetic
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import Descriptors

# -----------------------------------------------------------------------------
# Default real/syn pairing (ONBOARDING.md §1.3 -- "Synthetic counterparts").
# Split suffixes per the dataset catalogue table (all "_both" for these
# specific real-* sets). Override with --pairs-json if your configs differ.
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

# Same fixed property panel used elsewhere in this pipeline.


def _aromatic_fraction(mol) -> float:
    if mol.GetNumAtoms() == 0:
        return 0.0
    return sum(1 for a in mol.GetAtoms() if a.GetIsAromatic()) / mol.GetNumAtoms()


PROPERTIES = {
    "MolWt": Descriptors.MolWt,
    "LogP": Descriptors.MolLogP,
    "TPSA": Descriptors.TPSA,
    "RingCount": Descriptors.RingCount,
    "Aromaticity": _aromatic_fraction,
}

QUALITATIVE_PALETTE = plt.cm.tab10(np.linspace(0, 1, 10))
DPI = 600


@dataclass
class PairSpec:
    real_name: str
    syn_name: str
    real_suffix: str
    syn_suffix: str


# -----------------------------------------------------------------------------
# Model + datamodule loading (same conventions as extract_cotrain_embeddings.py)
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
# Alignment (mol_idx primary, canonical-SMILES fallback)
# -----------------------------------------------------------------------------


def canonicalize(smi: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol is not None else None


def align_pair(real: Dict, syn: Dict) -> Tuple[np.ndarray, np.ndarray, str, int]:
    """Returns (real_indices, syn_indices, method, n_smiles_mismatches), with
    real_indices[i] and syn_indices[i] referring to the *same* molecule."""
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
# Properties
# -----------------------------------------------------------------------------


def _property_worker(smi: str) -> List[float]:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return [np.nan] * len(PROPERTIES)
    vals = []
    for fn in PROPERTIES.values():
        try:
            vals.append(float(fn(mol)))
        except Exception:
            vals.append(np.nan)
    return vals


def compute_property_matrix(smiles_list: List[str], n_workers: int) -> np.ndarray:
    if n_workers <= 1 or len(smiles_list) < 200:
        rows = [_property_worker(s) for s in smiles_list]
    else:
        with mp.Pool(n_workers) as pool:
            rows = list(pool.imap(_property_worker, smiles_list, chunksize=64))
    return np.array(rows, dtype=np.float64)


# -----------------------------------------------------------------------------
# Per-pair statistics (exact, full pairwise matrices -- these sets are small)
# -----------------------------------------------------------------------------


@dataclass
class PairResult:
    label: str
    n_matched: int
    n_smiles_mismatch: int
    align_method: str
    real_embedding: np.ndarray
    syn_embedding: np.ndarray
    real_smiles: List[str]
    paired_euclidean: np.ndarray
    background_euclidean: np.ndarray
    paired_cosine: np.ndarray
    background_cosine: np.ndarray
    top1_real_to_syn: float
    top5_real_to_syn: float
    top1_syn_to_real: float
    top5_syn_to_real: float
    property_matrix: np.ndarray


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a_n @ b_n.T


def euclidean_dist_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)


def topk_identity_accuracy(sim_matrix: np.ndarray, k: int) -> float:
    n = sim_matrix.shape[0]
    correct = 0
    for i in range(n):
        row = sim_matrix[i]
        rank = int((row > row[i]).sum())  # how many entries strictly beat the true match
        if rank < k:
            correct += 1
    return correct / n


def compute_pair_result(label: str, real: Dict, syn: Dict, n_prop_workers: int) -> Optional[PairResult]:
    real_idx, syn_idx, method, n_mismatch = align_pair(real, syn)
    n = len(real_idx)
    if n < 2:
        print(f"[warn] {label}: only {n} matched molecule(s) (alignment={method}), skipping.")
        return None
    if n_mismatch > 0:
        print(f"[WARNING] {label}: {n_mismatch}/{n} matched pairs have DIFFERENT "
              f"canonical SMILES between real and syn -- alignment may be wrong!")

    real_emb = real["global_cond"][real_idx]
    syn_emb = syn["global_cond"][syn_idx]
    real_smi = [real["smiles"][i] for i in real_idx]

    euclid = euclidean_dist_matrix(real_emb, syn_emb)
    cosine = cosine_sim_matrix(real_emb, syn_emb)

    diag_idx = np.arange(n)
    paired_euclid = euclid[diag_idx, diag_idx]
    paired_cosine = cosine[diag_idx, diag_idx]

    off_mask = ~np.eye(n, dtype=bool)
    background_euclid = euclid[off_mask]
    background_cosine = cosine[off_mask]

    top1_r2s = topk_identity_accuracy(cosine, 1)
    top5_r2s = topk_identity_accuracy(cosine, min(5, n))
    top1_s2r = topk_identity_accuracy(cosine.T, 1)
    top5_s2r = topk_identity_accuracy(cosine.T, min(5, n))

    prop_matrix = compute_property_matrix(real_smi, n_prop_workers)

    return PairResult(
        label=label, n_matched=n, n_smiles_mismatch=n_mismatch, align_method=method,
        real_embedding=real_emb, syn_embedding=syn_emb, real_smiles=real_smi,
        paired_euclidean=paired_euclid, background_euclidean=background_euclid,
        paired_cosine=paired_cosine, background_cosine=background_cosine,
        top1_real_to_syn=top1_r2s, top5_real_to_syn=top5_r2s,
        top1_syn_to_real=top1_s2r, top5_syn_to_real=top5_s2r,
        property_matrix=prop_matrix,
    )


# -----------------------------------------------------------------------------
# Plots (same visual conventions as the rest of the pipeline)
# -----------------------------------------------------------------------------


def plot_pca_drift(result: PairResult, out_path: Path) -> None:
    combined = np.concatenate([result.real_embedding, result.syn_embedding], axis=0)
    n = result.n_matched
    pcs = PCA(n_components=2).fit_transform(combined)
    real_2d, syn_2d = pcs[:n], pcs[n:]

    fig, ax = plt.subplots(figsize=(7, 7))
    for i in range(n):
        ax.plot([real_2d[i, 0], syn_2d[i, 0]], [real_2d[i, 1], syn_2d[i, 1]],
               color="lightgray", alpha=0.5, linewidth=0.8, zorder=1)

    ax.scatter(real_2d[:, 0], real_2d[:, 1], color=QUALITATIVE_PALETTE[0], s=40,
              alpha=0.85, linewidths=0, label="real", zorder=2)
    ax.scatter(syn_2d[:, 0], syn_2d[:, 1], color=QUALITATIVE_PALETTE[1], s=40,
              alpha=0.85, linewidths=0, label="synthetic", zorder=2)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"{result.label}: real vs. synthetic embedding drift (n={n})")
    ax.legend(frameon=True, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved {out_path}")


def plot_distance_histogram(result: PairResult, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(result.background_euclidean, bins=30, color="lightgray", alpha=0.7,
           density=True, label="background (mismatched pairs)")
    ax.hist(result.paired_euclidean, bins=30, color=QUALITATIVE_PALETTE[3], alpha=0.7,
           density=True, label="paired (same molecule)")
    ax.axvline(result.paired_euclidean.mean(), color=QUALITATIVE_PALETTE[3], linestyle="--", lw=1.2)
    ax.axvline(result.background_euclidean.mean(), color="gray", linestyle="--", lw=1.2)
    ax.set_xlabel("Euclidean distance (global_cond)")
    ax.set_ylabel("density")
    ax.set_title(f"{result.label}: paired vs. background distance")
    ax.legend(frameon=True, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved {out_path}")


def plot_property_correlation_grid(result: PairResult, property_names: List[str], out_path: Path) -> None:
    ncols = min(4, len(property_names))
    nrows = int(np.ceil(len(property_names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.8 * nrows),
                             constrained_layout=True, squeeze=False)

    for i in range(nrows * ncols):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        if i >= len(property_names):
            ax.axis("off")
            continue
        p_name = property_names[i]
        vals = result.property_matrix[:, i]
        mask = np.isfinite(vals)
        if mask.sum() < 3:
            ax.axis("off")
            continue
        r_val, _ = pearsonr(vals[mask], result.paired_euclidean[mask])
        ax.scatter(vals[mask], result.paired_euclidean[mask], alpha=0.5, s=10, linewidths=0)
        ax.set_title(f"{p_name}\nr={r_val:.3f}", fontsize=9)
        ax.set_xlabel(p_name)
        ax.set_ylabel("paired distance")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(f"{result.label}: does the real/synthetic gap track molecular properties?", fontsize=10)
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved {out_path}")


def plot_retrieval_accuracy_bar(results: List[PairResult], out_path: Path) -> None:
    labels = [r.label for r in results]
    x = np.arange(len(labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(labels)), 4.5))
    ax.bar(x - 1.5 * width, [r.top1_real_to_syn for r in results], width, label="top1 real\u2192syn", color=QUALITATIVE_PALETTE[0])
    ax.bar(x - 0.5 * width, [r.top5_real_to_syn for r in results], width, label="top5 real\u2192syn", color=QUALITATIVE_PALETTE[1])
    ax.bar(x + 0.5 * width, [r.top1_syn_to_real for r in results], width, label="top1 syn\u2192real", color=QUALITATIVE_PALETTE[2])
    ax.bar(x + 1.5 * width, [r.top5_syn_to_real for r in results], width, label="top5 syn\u2192real", color=QUALITATIVE_PALETTE[3])

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("identity retrieval accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Cross-spectra-source molecule identity retrieval")
    ax.legend(frameon=True, fontsize=8, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved {out_path}")


# -----------------------------------------------------------------------------
# Report
# -----------------------------------------------------------------------------


def write_report(results: List[PairResult], property_names: List[str], out_path: Path) -> None:
    lines = [
        "Real vs. synthetic spectra embedding-gap report",
        "=" * 72,
        "",
        "gap ratio = mean(paired real-vs-syn distance) / mean(background "
        "mismatched-pair distance). << 1 means the model still tightly "
        "recognizes molecular identity despite the spectra-source switch; "
        "values approaching or exceeding 1 mean spectra source rivals "
        "molecular identity as a driver of the embedding.",
        "",
    ]

    for res in results:
        gap_ratio = float(res.paired_euclidean.mean() / (res.background_euclidean.mean() + 1e-12))
        cos_gap = float(res.paired_cosine.mean() - res.background_cosine.mean())

        lines.append(f"--- {res.label} (n={res.n_matched}, alignment={res.align_method}) ---")
        if res.n_smiles_mismatch > 0:
            lines.append(f"  !! {res.n_smiles_mismatch} SMILES MISMATCHES in matched pairs -- "
                         f"check alignment / split suffix config for this pair.")
        lines.append(f"  mean paired Euclidean distance      : {res.paired_euclidean.mean():.4f} "
                     f"(median {np.median(res.paired_euclidean):.4f})")
        lines.append(f"  mean background Euclidean distance  : {res.background_euclidean.mean():.4f}")
        lines.append(f"  gap ratio (paired / background)     : {gap_ratio:.3f}")
        lines.append(f"  mean paired cosine similarity        : {res.paired_cosine.mean():.4f}")
        lines.append(f"  mean background cosine similarity    : {res.background_cosine.mean():.4f}")
        lines.append(f"  cosine similarity gap (paired-bg)    : {cos_gap:.4f}")
        lines.append(f"  identity retrieval real\u2192syn  top1/top5: {res.top1_real_to_syn:.3f} / {res.top5_real_to_syn:.3f}")
        lines.append(f"  identity retrieval syn\u2192real  top1/top5: {res.top1_syn_to_real:.3f} / {res.top5_syn_to_real:.3f}")

        for p_idx, p_name in enumerate(property_names):
            vals = res.property_matrix[:, p_idx]
            mask = np.isfinite(vals)
            if mask.sum() < 3:
                continue
            r_val, p_val = pearsonr(vals[mask], res.paired_euclidean[mask])
            lines.append(f"  corr(paired distance, {p_name}): r={r_val:.3f}, p={p_val:.3g}")
        lines.append("")

    # Pooled stats across all pairs
    all_paired = np.concatenate([r.paired_euclidean for r in results])
    all_background = np.concatenate([r.background_euclidean for r in results])
    all_top1_r2s = np.mean([r.top1_real_to_syn for r in results])
    all_top1_s2r = np.mean([r.top1_syn_to_real for r in results])

    lines.append("--- POOLED across all pairs ---")
    lines.append(f"  mean paired Euclidean distance     : {all_paired.mean():.4f}")
    lines.append(f"  mean background Euclidean distance : {all_background.mean():.4f}")
    lines.append(f"  gap ratio (paired / background)    : {all_paired.mean() / (all_background.mean() + 1e-12):.3f}")
    lines.append(f"  mean top1 real\u2192syn retrieval accuracy (unweighted across pairs): {all_top1_r2s:.3f}")
    lines.append(f"  mean top1 syn\u2192real retrieval accuracy (unweighted across pairs): {all_top1_s2r:.3f}")

    out_path.write_text("\n".join(lines))
    print(f"✓ Saved {out_path}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Real vs. synthetic spectra embedding-gap analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--nmr3d-root", type=Path, required=True)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--ckpt-sigma", type=float, required=True)
    p.add_argument("--condition", type=str, default="hcpeak")
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"],
                   help="These are small eval-only benchmark sets; 'test' is "
                        "the usual choice unless your split config differs.")
    p.add_argument("--pairs-json", type=Path, default=None,
                   help="Override the default real-*/syn-* pairing list "
                        "(JSON list of {real_name, syn_name, real_suffix, syn_suffix}).")
    p.add_argument("--out-dir", type=Path, default=Path("real_vs_synthetic_summary"))
    p.add_argument("--n-prop-workers", type=int, default=max(1, mp.cpu_count() - 2))
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

    if args.pairs_json is not None:
        pairs_raw = json.loads(Path(args.pairs_json).read_text())
    else:
        pairs_raw = DEFAULT_PAIRS
    pairs = [PairSpec(**p) for p in pairs_raw]

    print("=" * 78)
    print("Real vs. synthetic spectra embedding-gap analysis")
    print(f"  ckpt   : {args.ckpt}")
    print(f"  sigma  : {args.ckpt_sigma}")
    print(f"  pairs  : {[(p.real_name, p.syn_name) for p in pairs]}")
    print("=" * 78)

    model, peak_embedder = load_model(args.ckpt, args.device)
    property_names = list(PROPERTIES.keys())

    results: List[PairResult] = []
    for pair in pairs:
        label = f"{pair.real_name}_vs_{pair.syn_name}"
        print(f"\n### {label} ###")

        try:
            real_dm = build_datamodule(config_dir, pair.real_name, pair.real_suffix, args.ckpt_sigma, args.condition)
            real_data = extract_split_embeddings(model, peak_embedder, real_dm, args.split, args.device)

            syn_dm = build_datamodule(config_dir, pair.syn_name, pair.syn_suffix, args.ckpt_sigma, args.condition)
            syn_data = extract_split_embeddings(model, peak_embedder, syn_dm, args.split, args.device)
        except Exception as e:
            print(f"[warn] Failed to load/extract {label}: {e}. Skipping this pair.")
            continue

        result = compute_pair_result(label, real_data, syn_data, args.n_prop_workers)
        if result is None:
            continue
        results.append(result)

        plot_pca_drift(result, args.out_dir / f"{label}_pca_drift.png")
        plot_distance_histogram(result, args.out_dir / f"{label}_distance_hist.png")
        plot_property_correlation_grid(result, property_names, args.out_dir / f"{label}_property_correlation.png")

    if not results:
        print("\n[warn] No pairs produced usable results -- nothing to report.")
        return

    plot_retrieval_accuracy_bar(results, args.out_dir / "retrieval_accuracy_by_pair.png")
    write_report(results, property_names, args.out_dir / "real_vs_synthetic_gap_report.txt")

    print("\nDone.")


if __name__ == "__main__":
    main()
