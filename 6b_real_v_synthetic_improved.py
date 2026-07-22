#!/usr/bin/env python
"""
6b_real_v_synthetic_improved.py
=========================

nohup python 6b_real_v_synthetic_improved.py --nmr3d-root /home/jc4587/3_AI4chemistr/nmr-to-3d --ckpt /projects/CRYOEM/zhonglab/data_nmr/2026/ckpts/26-05-01-cotraining-baselines/cotrain-epoch0899-accuracy60_70.ckpt --ckpt-sigma 2.8268 --cotrain-data-dir /scratch/gpfs/ZHONGE/jc4587/nmr_embs_FINAL --h-shift-key h_peak_centroid --c-shift-key c_peak_centroid --h-mask-key h_peak_mask --c-mask-key c_peak_mask --h-nh-key h_peak_nH --h-multiplet-key h_peak_multiplet --out-dir 6b_real_vs_synthetic_improved > 6b_real_v_synthetic_improved.out


Merges the former `real_vs_synthetic_gap.py` (statistics: gap ratio,
retrieval accuracy, property correlation) and `6a_real_v_synthetic_umap.py`
(UMAP-in-context visualization) into a single script/pipeline, with the
following changes from both predecessors:

    1. COSINE, NOT EUCLIDEAN. Every "distance" used for the gap ratio, the
       paired-vs-background comparison, and the correlation-vs-distance
       analysis is now a cosine distance (1 - cosine similarity), not
       Euclidean. (Retrieval accuracy was already cosine-similarity-based
       in the original script and is unchanged.) UMAP itself also defaults
       to metric="cosine".

    2. SPECTRAL feature correlation, not molecular-property correlation.
       Correlates paired distance against features of the INPUT NMR
       SPECTRUM itself: chemical-shift distribution/crowding for H and C,
       plus (if available) attached-proton count (h_peak_nH) and
       multiplet class (h_peak_multiplet) per H peak.

    3. ONE extraction pass per pair (embeddings + raw condition tensors
       together), instead of the old two-script setup re-running the
       forward pass twice.

    4. Per-pair standalone PCA "drift" plots are DROPPED (redundant with
       the UMAP-in-context main figure + per-pair insets).

    5. No rectangle boxes around each pair's cluster in the main figure.

------------------------------------------------------------------------
IMPORTANT CAVEAT: SPECTRAL FEATURE EXTRACTION DEPENDS ON YOUR SCHEMA
------------------------------------------------------------------------
Default key names (--h-shift-key/--c-shift-key/--h-mask-key/--c-mask-key)
are guesses. If the configured keys aren't found in the condition dict for
a given pair, this script prints the ACTUAL keys it found and skips
spectral-feature correlation for that pair with a warning, rather than
crashing or silently computing nonsense.

Confirmed working key names for this project's "hcpeak" condition type
(from a real run):
    --h-shift-key h_peak_centroid --c-shift-key c_peak_centroid
    --h-mask-key h_peak_mask      --c-mask-key c_peak_mask
    --h-nh-key h_peak_nH          --h-multiplet-key h_peak_multiplet
------------------------------------------------------------------------

Usage
-----
python 6b_real_v_synthetic_improved.py --nmr3d-root /home/jc4587/3_AI4chemistr/nmr-to-3d --ckpt /projects/CRYOEM/zhonglab/data_nmr/2026/ckpts/26-07-17-cotraining-clean-v2/cotrain-clean-v2-epoch0349-accuracy68_06.ckpt --ckpt-sigma 2.8268 --cotrain-data-dir /scratch/gpfs/ZHONGE/jc4587/nmr_embs_FINAL --h-shift-key h_peak_centroid --c-shift-key c_peak_centroid --h-mask-key h_peak_mask --c-mask-key c_peak_mask --h-nh-key h_peak_nH --h-multiplet-key h_peak_multiplet --out-dir 6b_real_vs_synthetic_improved
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
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr, skew, kurtosis
from tqdm import tqdm

from rdkit import Chem, RDLogger
from rdkit.Chem import Draw

RDLogger.DisableLog("rdApp.*")

QUALITATIVE_PALETTE = plt.cm.tab10(np.linspace(0, 1, 10))
DPI = 600

# -----------------------------------------------------------------------------
# Default real/syn pairing (ONBOARDING.md §1.3 -- "Synthetic counterparts")
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
# Model + datamodule loading (unchanged from both predecessors)
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


# -----------------------------------------------------------------------------
# Extraction: embeddings + smiles + mol_idx + RAW condition tensors (for
# spectral features), all in ONE forward pass over the dataloader.
# -----------------------------------------------------------------------------

def extract_split_data(model, peak_embedder, dm, split: str, device: str) -> Dict:
    dataloader = get_split_dataloader(dm, split)

    smiles_out: List[str] = []
    mol_idx_out: List[int] = []
    global_cond_out: List[torch.Tensor] = []
    condition_out: Dict[str, List[torch.Tensor]] = {}
    have_mol_idx = True

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"    forward pass ({split})"):
            model_inputs, smiles = batch[0]
            condition = {k: v.to(device) for k, v in model_inputs["condition"].items()}
            global_cond, _, _, _ = peak_embedder(condition, extract_all=True)

            global_cond_out.append(global_cond.cpu())
            smiles_out.extend(smiles)

            for k, v in condition.items():
                condition_out.setdefault(k, []).append(v.detach().cpu())

            if have_mol_idx:
                mol_idx_batch = model_inputs.get("mol_idx") if isinstance(model_inputs, dict) else None
                if mol_idx_batch is not None:
                    mol_idx_out.extend(mol_idx_batch.detach().cpu().tolist())
                else:
                    have_mol_idx = False
                    mol_idx_out = []

    condition_cat = {}
    for k, parts in condition_out.items():
        try:
            condition_cat[k] = torch.cat(parts, dim=0).numpy()
        except Exception:
            condition_cat[k] = [p for part in parts for p in part]

    return {
        "smiles": smiles_out,
        "global_cond": torch.cat(global_cond_out, dim=0).numpy().astype(np.float32),
        "mol_idx": np.array(mol_idx_out, dtype=np.int64) if have_mol_idx and len(mol_idx_out) == len(smiles_out) else None,
        "condition": condition_cat,
    }


# -----------------------------------------------------------------------------
# Alignment (mol_idx primary, canonical-SMILES fallback -- unchanged)
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
# Cosine similarity / distance helpers (replaces Euclidean throughout)
# -----------------------------------------------------------------------------

def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a_n @ b_n.T


def paired_cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return 1.0 - np.sum(a_n * b_n, axis=1)


def topk_identity_accuracy(sim_matrix: np.ndarray, k: int) -> float:
    n = sim_matrix.shape[0]
    correct = 0
    for i in range(n):
        row = sim_matrix[i]
        rank = int((row > row[i]).sum())
        if rank < k:
            correct += 1
    return correct / n


# -----------------------------------------------------------------------------
# Spectral feature extraction (SEE "IMPORTANT CAVEAT" IN THE MODULE DOCSTRING)
# -----------------------------------------------------------------------------

SPECTRAL_FEATURE_NAMES = [
    "n_H_peaks", "H_shift_min", "H_shift_max", "H_shift_range", "H_shift_mean",
    "H_shift_median", "H_shift_std", "H_shift_skew", "H_shift_kurtosis",
    "H_mean_peak_spacing", "H_min_peak_spacing", "H_peaks_per_ppm",
    "n_C_peaks", "C_shift_min", "C_shift_max", "C_shift_range", "C_shift_mean",
    "C_shift_median", "C_shift_std", "C_shift_skew", "C_shift_kurtosis",
    "C_mean_peak_spacing", "C_min_peak_spacing", "C_peaks_per_ppm",
    "H_to_C_peak_ratio", "total_peaks",
]


def _one_axis_features(prefix: str, values: np.ndarray) -> Dict[str, float]:
    n = len(values)
    out = {f"{prefix}_shift_min": np.nan, f"{prefix}_shift_max": np.nan,
           f"{prefix}_shift_range": np.nan, f"{prefix}_shift_mean": np.nan,
           f"{prefix}_shift_median": np.nan, f"{prefix}_shift_std": np.nan,
           f"{prefix}_shift_skew": np.nan, f"{prefix}_shift_kurtosis": np.nan,
           f"{prefix}_mean_peak_spacing": np.nan, f"{prefix}_min_peak_spacing": np.nan,
           f"{prefix}_peaks_per_ppm": np.nan}
    out[f"n_{prefix}_peaks"] = n
    if n == 0:
        return out
    v = np.sort(values)
    shift_range = float(v[-1] - v[0])
    out[f"{prefix}_shift_min"] = float(v[0])
    out[f"{prefix}_shift_max"] = float(v[-1])
    out[f"{prefix}_shift_range"] = shift_range
    out[f"{prefix}_shift_mean"] = float(v.mean())
    out[f"{prefix}_shift_median"] = float(np.median(v))
    if n > 1:
        out[f"{prefix}_shift_std"] = float(v.std(ddof=1))
        gaps = np.diff(v)
        out[f"{prefix}_mean_peak_spacing"] = float(gaps.mean())
        out[f"{prefix}_min_peak_spacing"] = float(gaps.min())
        if shift_range > 0:
            out[f"{prefix}_peaks_per_ppm"] = n / shift_range
    if n > 2:
        out[f"{prefix}_shift_skew"] = float(skew(v))
        out[f"{prefix}_shift_kurtosis"] = float(kurtosis(v))
    return out


def _multiplet_features(values: np.ndarray) -> Dict[str, float]:
    """Treats h_peak_multiplet as an unknown categorical encoding (no
    assumption about what each class code means): how many distinct
    classes appear, how concentrated peaks are in the single most common
    class, and the Shannon entropy of the class distribution."""
    n = len(values)
    out = {"H_multiplet_n_unique": np.nan, "H_multiplet_mode_frac": np.nan,
           "H_multiplet_entropy": np.nan}
    if n == 0:
        return out
    _, counts = np.unique(values, return_counts=True)
    probs = counts / n
    out["H_multiplet_n_unique"] = float(len(counts))
    out["H_multiplet_mode_frac"] = float(counts.max() / n)
    out["H_multiplet_entropy"] = float(-(probs * np.log(probs + 1e-12)).sum())
    return out


def _nh_features(values: np.ndarray) -> Dict[str, float]:
    """h_peak_nH = attached-proton count per H peak (1/2/3 for CH/CH2/CH3)."""
    n = len(values)
    out = {"H_nH_mean": np.nan, "H_nH_max": np.nan, "H_nH_sum": np.nan,
           "H_frac_CH": np.nan, "H_frac_CH2": np.nan, "H_frac_CH3": np.nan}
    if n == 0:
        return out
    out["H_nH_mean"] = float(values.mean())
    out["H_nH_max"] = float(values.max())
    out["H_nH_sum"] = float(values.sum())
    out["H_frac_CH"] = float(np.mean(values == 1))
    out["H_frac_CH2"] = float(np.mean(values == 2))
    out["H_frac_CH3"] = float(np.mean(values == 3))
    return out


def compute_spectral_feature_matrix(condition: Dict[str, np.ndarray], idx: np.ndarray,
                                     h_shift_key: str, c_shift_key: str,
                                     h_mask_key: Optional[str], c_mask_key: Optional[str],
                                     h_nh_key: Optional[str] = None,
                                     h_multiplet_key: Optional[str] = None,
                                     ) -> Tuple[Optional[np.ndarray], List[str]]:
    """Returns (feature_matrix [n_matched, n_features], feature_names), or
    (None, []) if the configured shift keys aren't present in `condition`.
    feature_names is DYNAMIC: it's the base shift-derived feature list, plus
    nH-derived features if h_nh_key is given and found, plus multiplet-
    derived features if h_multiplet_key is given and found."""
    missing = [k for k in (h_shift_key, c_shift_key) if k not in condition]
    if missing:
        print(f"[warn] Spectral feature keys not found in condition dict: {missing}. "
              f"Available keys: {sorted(condition.keys())}. "
              f"Pass the correct --h-shift-key/--c-shift-key (and mask keys, if "
              f"applicable) to enable spectral-feature correlation for this pair. "
              f"Skipping spectral correlation for this pair.")
        return None, []

    h_shifts_all = condition[h_shift_key][idx]
    c_shifts_all = condition[c_shift_key][idx]
    h_mask_all = condition[h_mask_key][idx] if (h_mask_key and h_mask_key in condition) else None
    c_mask_all = condition[c_mask_key][idx] if (c_mask_key and c_mask_key in condition) else None
    h_nh_all = condition[h_nh_key][idx] if (h_nh_key and h_nh_key in condition) else None
    h_mult_all = condition[h_multiplet_key][idx] if (h_multiplet_key and h_multiplet_key in condition) else None

    feature_names = list(SPECTRAL_FEATURE_NAMES)
    if h_nh_all is not None:
        feature_names += ["H_nH_mean", "H_nH_max", "H_nH_sum", "H_frac_CH", "H_frac_CH2", "H_frac_CH3"]
    if h_mult_all is not None:
        feature_names += ["H_multiplet_n_unique", "H_multiplet_mode_frac", "H_multiplet_entropy"]

    rows = []
    for i in range(len(idx)):
        h_mask_i = h_mask_all[i].astype(bool) if h_mask_all is not None else np.isfinite(h_shifts_all[i])
        c_mask_i = c_mask_all[i].astype(bool) if c_mask_all is not None else np.isfinite(c_shifts_all[i])
        h_valid = h_shifts_all[i][h_mask_i]
        c_valid = c_shifts_all[i][c_mask_i]

        feats = {}
        feats.update(_one_axis_features("H", h_valid))
        feats.update(_one_axis_features("C", c_valid))
        n_h, n_c = feats["n_H_peaks"], feats["n_C_peaks"]
        feats["H_to_C_peak_ratio"] = float(n_h / n_c) if n_c > 0 else np.nan
        feats["total_peaks"] = float(n_h + n_c)

        row = [feats[name] for name in SPECTRAL_FEATURE_NAMES]

        if h_nh_all is not None:
            nh_valid = h_nh_all[i][h_mask_i]
            nh_feats = _nh_features(nh_valid)
            row += [nh_feats["H_nH_mean"], nh_feats["H_nH_max"], nh_feats["H_nH_sum"],
                    nh_feats["H_frac_CH"], nh_feats["H_frac_CH2"], nh_feats["H_frac_CH3"]]
        if h_mult_all is not None:
            mult_valid = h_mult_all[i][h_mask_i]
            mult_feats = _multiplet_features(mult_valid)
            row += [mult_feats["H_multiplet_n_unique"], mult_feats["H_multiplet_mode_frac"],
                    mult_feats["H_multiplet_entropy"]]

        rows.append(row)

    return np.array(rows, dtype=np.float64), feature_names


# -----------------------------------------------------------------------------
# UMAP: fit (or load) on the cotrain background.
# -----------------------------------------------------------------------------

class FittedProjector:
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


def fit_or_load_umap(embedding: np.ndarray, n_neighbors: int, min_dist: float,
                      metric: str, seed: int, save_model_path: Optional[Path],
                      load_model_path: Optional[Path], pca_components: Optional[int],
                      use_scaler: bool) -> Tuple["FittedProjector", np.ndarray]:
    if load_model_path is not None:
        print(f"Loading fitted projector (scaler/PCA/UMAP) from {load_model_path}")
        with open(load_model_path, "rb") as f:
            projector = pickle.load(f)
        if not isinstance(projector, FittedProjector):
            raise TypeError(f"{load_model_path} does not contain a FittedProjector.")
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

    print(f"Fitting UMAP (metric={metric}, n_neighbors={n_neighbors}, "
          f"min_dist={min_dist}) on {x.shape[0]} points ...")
    umap_reducer = umap_lib.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                                  metric=metric, random_state=seed)
    emb2d = umap_reducer.fit_transform(x)

    projector = FittedProjector(scaler, pca, umap_reducer)

    if save_model_path is not None:
        save_model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_model_path, "wb") as f:
            pickle.dump(projector, f)
        print(f"Cached fitted projector to {save_model_path}")

    return projector, emb2d


# -----------------------------------------------------------------------------
# Per-pair result
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
    paired_cos_dist: np.ndarray
    background_cos_dist: np.ndarray
    paired_cos_sim: np.ndarray
    top1_real_to_syn: float
    top5_real_to_syn: float
    top1_syn_to_real: float
    top5_syn_to_real: float
    spectral_features: Optional[np.ndarray]
    feature_names: List[str]
    color: np.ndarray
    center: Tuple[float, float]
    half_extent: float


def compute_pair_result(label: str, real: Dict, syn: Dict, reducer, color: np.ndarray,
                         window_size: Optional[float], min_window: float, padding_frac: float,
                         h_shift_key: str, c_shift_key: str,
                         h_mask_key: Optional[str], c_mask_key: Optional[str],
                         h_nh_key: Optional[str] = None,
                         h_multiplet_key: Optional[str] = None) -> Optional[PairResult]:
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

    paired_cos_dist = paired_cosine_distance(real_emb, syn_emb)
    cosine_full = cosine_sim_matrix(real_emb, syn_emb)
    paired_cos_sim = np.diag(cosine_full)

    if n >= 2:
        off_mask = ~np.eye(n, dtype=bool)
        background_cos_dist = (1.0 - cosine_full)[off_mask]
    else:
        background_cos_dist = np.array([])

    top1_r2s = topk_identity_accuracy(cosine_full, 1)
    top5_r2s = topk_identity_accuracy(cosine_full, min(5, n))
    top1_s2r = topk_identity_accuracy(cosine_full.T, 1)
    top5_s2r = topk_identity_accuracy(cosine_full.T, min(5, n))

    spectral_features, feature_names = compute_spectral_feature_matrix(
        real["condition"], real_idx, h_shift_key, c_shift_key, h_mask_key, c_mask_key,
        h_nh_key, h_multiplet_key,
    )

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
        paired_cos_dist=paired_cos_dist, background_cos_dist=background_cos_dist,
        paired_cos_sim=paired_cos_sim,
        top1_real_to_syn=top1_r2s, top5_real_to_syn=top5_r2s,
        top1_syn_to_real=top1_s2r, top5_syn_to_real=top5_s2r,
        spectral_features=spectral_features, feature_names=feature_names,
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
    print(f"Saved {out_path}")


def mol_grid_for_top_shifted(pr: PairResult, n_mols: int):
    order = np.argsort(-pr.paired_cos_dist)[:n_mols]
    mols, legends = [], []
    for idx in order:
        mol = Chem.MolFromSmiles(pr.matched_smiles[idx])
        if mol is None:
            continue
        mols.append(mol)
        legends.append(f"cos_dist={pr.paired_cos_dist[idx]:.3f}")
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
    ax_mols.set_title("most-shifted molecules (real vs synthetic, cosine distance)")

    plt.tight_layout()
    out_path = out_dir / f"inset_{pr.label}.png"
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    if mol_img is not None:
        mol_img.save(out_dir / f"inset_{pr.label}_mols.png")

    print(f"Saved {out_path}")


def plot_distance_histogram(pr: PairResult, out_path: Path, dpi: int) -> None:
    if len(pr.background_cos_dist) == 0:
        print(f"[warn] {pr.label}: not enough points for a background distribution, skipping histogram.")
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(pr.background_cos_dist, bins=30, color="lightgray", alpha=0.7,
            density=True, label="background (mismatched pairs)")
    ax.hist(pr.paired_cos_dist, bins=30, color=QUALITATIVE_PALETTE[3], alpha=0.7,
            density=True, label="paired (same molecule)")
    ax.axvline(pr.paired_cos_dist.mean(), color=QUALITATIVE_PALETTE[3], linestyle="--", lw=1.2)
    ax.axvline(pr.background_cos_dist.mean(), color="gray", linestyle="--", lw=1.2)
    ax.set_xlabel("cosine distance (1 - cosine similarity), global_cond")
    ax.set_ylabel("density")
    ax.set_title(f"{pr.label}: paired vs. background cosine distance")
    ax.legend(frameon=True, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_spectral_correlation_grid(pr: PairResult, out_path: Path, dpi: int) -> None:
    if pr.spectral_features is None:
        print(f"[warn] {pr.label}: no spectral features available, skipping correlation plot.")
        return
    feature_names = pr.feature_names
    ncols = 5
    nrows = int(np.ceil(len(feature_names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.4 * ncols, 2.6 * nrows),
                              constrained_layout=True, squeeze=False)

    for i in range(nrows * ncols):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        if i >= len(feature_names):
            ax.axis("off")
            continue
        f_name = feature_names[i]
        vals = pr.spectral_features[:, i]
        mask = np.isfinite(vals) & np.isfinite(pr.paired_cos_dist)
        if mask.sum() < 3 or np.std(vals[mask]) == 0:
            ax.axis("off")
            continue
        r_val, _ = pearsonr(vals[mask], pr.paired_cos_dist[mask])
        ax.scatter(vals[mask], pr.paired_cos_dist[mask], alpha=0.5, s=8, linewidths=0)
        ax.set_title(f"{f_name}\nr={r_val:.3f}", fontsize=8)
        ax.set_xlabel(f_name, fontsize=7)
        ax.set_ylabel("paired cos. dist.", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(f"{pr.label}: does the real/synthetic gap track input-spectrum features?", fontsize=10)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_retrieval_accuracy_bar(results: List[PairResult], out_path: Path, dpi: int) -> None:
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
    ax.set_ylabel("identity retrieval accuracy (cosine similarity)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Cross-spectra-source molecule identity retrieval")
    ax.legend(frameon=True, fontsize=8, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# -----------------------------------------------------------------------------
# Report
# -----------------------------------------------------------------------------

def write_report(results: List[PairResult], out_path: Path) -> None:
    lines = [
        "Real vs. synthetic spectra: embedding-gap report (cosine distance)",
        "=" * 72,
        "",
        "gap ratio = mean(paired cosine distance) / mean(background cosine "
        "distance). << 1 means the model still tightly recognizes molecular "
        "identity despite the spectra-source switch; values approaching or "
        "exceeding 1 mean spectra source rivals molecular identity as a "
        "driver of the embedding.",
        "",
    ]

    for res in results:
        gap_ratio = (
            float(res.paired_cos_dist.mean() / (res.background_cos_dist.mean() + 1e-12))
            if len(res.background_cos_dist) else float("nan")
        )

        lines.append(f"--- {res.label} (n={res.n_matched}, alignment={res.align_method}) ---")
        if res.n_smiles_mismatch > 0:
            lines.append(f"  !! {res.n_smiles_mismatch} SMILES MISMATCHES in matched pairs -- "
                         f"check alignment / split suffix config for this pair.")
        lines.append(f"  mean paired cosine distance         : {res.paired_cos_dist.mean():.4f} "
                     f"(median {np.median(res.paired_cos_dist):.4f})")
        if len(res.background_cos_dist):
            lines.append(f"  mean background cosine distance     : {res.background_cos_dist.mean():.4f}")
            lines.append(f"  gap ratio (paired / background)     : {gap_ratio:.3f}")
        lines.append(f"  mean paired cosine similarity        : {res.paired_cos_sim.mean():.4f}")
        lines.append(f"  identity retrieval real\u2192syn  top1/top5: {res.top1_real_to_syn:.3f} / {res.top5_real_to_syn:.3f}")
        lines.append(f"  identity retrieval syn\u2192real  top1/top5: {res.top1_syn_to_real:.3f} / {res.top5_syn_to_real:.3f}")

        if res.spectral_features is not None:
            lines.append("  spectral feature correlations (vs. paired cosine distance):")
            for f_idx, f_name in enumerate(res.feature_names):
                vals = res.spectral_features[:, f_idx]
                mask = np.isfinite(vals) & np.isfinite(res.paired_cos_dist)
                if mask.sum() < 3 or np.std(vals[mask]) == 0:
                    continue
                r_val, p_val = pearsonr(vals[mask], res.paired_cos_dist[mask])
                lines.append(f"    {f_name:26s}: r={r_val:+.3f}, p={p_val:.3g}")
        else:
            lines.append("  spectral feature correlations: SKIPPED (condition keys not found -- see warnings above)")
        lines.append("")

    all_paired = np.concatenate([r.paired_cos_dist for r in results])
    all_background = np.concatenate([r.background_cos_dist for r in results if len(r.background_cos_dist)])
    all_top1_r2s = np.mean([r.top1_real_to_syn for r in results])
    all_top1_s2r = np.mean([r.top1_syn_to_real for r in results])

    lines.append("--- POOLED across all pairs ---")
    lines.append(f"  mean paired cosine distance     : {all_paired.mean():.4f}")
    if len(all_background):
        lines.append(f"  mean background cosine distance : {all_background.mean():.4f}")
        lines.append(f"  gap ratio (paired / background) : {all_paired.mean() / (all_background.mean() + 1e-12):.3f}")
    lines.append(f"  mean top1 real\u2192syn retrieval accuracy (unweighted across pairs): {all_top1_r2s:.3f}")
    lines.append(f"  mean top1 syn\u2192real retrieval accuracy (unweighted across pairs): {all_top1_s2r:.3f}")

    out_path.write_text("\n".join(lines))
    print(f"Saved {out_path}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merged real-vs-synthetic spectra embedding-gap analysis + "
                    "UMAP-in-context visualization (cosine distance throughout).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--nmr3d-root", type=Path, required=True)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--ckpt-sigma", type=float, required=True)
    p.add_argument("--condition", type=str, default="hcpeak")
    p.add_argument("--real-syn-split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--pairs-json", type=Path, default=None)

    p.add_argument("--cotrain-data-dir", type=Path, required=True)
    p.add_argument("--cotrain-prefix", type=str, default="cotrain")
    p.add_argument("--cotrain-splits", nargs="+", default=["train"], choices=["train", "val", "test"])
    p.add_argument("--embedding-key", type=str, default="global_cond")

    p.add_argument("--umap-n-neighbors", type=int, default=30)
    p.add_argument("--umap-min-dist", type=float, default=0.1)
    p.add_argument("--umap-metric", type=str, default="cosine")
    p.add_argument("--pca-components", type=int, default=50)
    p.add_argument("--no-scale", action="store_true")
    p.add_argument("--save-umap-model", type=Path, default=None)
    p.add_argument("--load-umap-model", type=Path, default=None)

    p.add_argument("--window-size", type=float, default=None)
    p.add_argument("--min-window-size", type=float, default=1.5)
    p.add_argument("--window-padding-frac", type=float, default=0.35)
    p.add_argument("--n-mols-per-inset", type=int, default=6)

    p.add_argument("--h-shift-key", type=str, default="h_peak_centroid")
    p.add_argument("--c-shift-key", type=str, default="c_peak_centroid")
    p.add_argument("--h-mask-key", type=str, default="h_peak_mask")
    p.add_argument("--c-mask-key", type=str, default="c_peak_mask")
    p.add_argument("--h-nh-key", type=str, default=None,
                   help="Optional key for attached-proton count per H peak "
                        "(e.g. 'h_peak_nH'). Adds CH/CH2/CH3 composition features.")
    p.add_argument("--h-multiplet-key", type=str, default=None,
                   help="Optional key for multiplet class per H peak "
                        "(e.g. 'h_peak_multiplet', treated as an unknown categorical "
                        "code -- adds entropy/uniqueness features, not semantic ones).")

    p.add_argument("--out-dir", type=Path, default=Path("6_real_vs_synthetic"))
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
    print("Real vs. synthetic spectra: merged analysis + visualization (cosine distance)")
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
            real_data = extract_split_data(model, peak_embedder, real_dm, args.real_syn_split, args.device)

            syn_dm = build_datamodule(config_dir, pair.syn_name, pair.syn_suffix, args.ckpt_sigma, args.condition)
            syn_data = extract_split_data(model, peak_embedder, syn_dm, args.real_syn_split, args.device)
        except Exception as e:
            print(f"[warn] Failed to load/extract {label}: {e}. Skipping this pair.")
            continue

        result = compute_pair_result(
            label, real_data, syn_data, reducer, color,
            args.window_size, args.min_window_size, args.window_padding_frac,
            args.h_shift_key, args.c_shift_key, args.h_mask_key, args.c_mask_key,
            args.h_nh_key, args.h_multiplet_key,
        )
        if result is None:
            continue
        pair_results.append(result)

        plot_distance_histogram(result, args.out_dir / f"{label}_distance_hist.png", args.dpi)
        plot_spectral_correlation_grid(result, args.out_dir / f"{label}_spectral_correlation.png", args.dpi)
        plot_pair_inset(result, cotrain_emb2d, args.n_mols_per_inset, args.out_dir, args.dpi)

    if not pair_results:
        print("\n[warn] No pairs produced usable results -- nothing to plot/report.")
        return

    plot_main_scatter(cotrain_emb2d, pair_results, args.out_dir / "umap_real_vs_synthetic_main.png", args.dpi)
    plot_retrieval_accuracy_bar(pair_results, args.out_dir / "retrieval_accuracy_by_pair.png", args.dpi)
    write_report(pair_results, args.out_dir / "real_vs_synthetic_report.txt")

    print("\nDone.")


if __name__ == "__main__":
    main()