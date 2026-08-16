"""Per-representation metrics for the decoder layer-comparison analysis.

Port of the previous pipeline's `src/analysis/layer_comparison_metrics.py`
(itself restored from `legacy_archive/8_layer_comparison.py`): property
linear-decodability, unsupervised cluster-quality metrics, and ECFP-vs-embedding
nearest-neighbor agreement. Numerically unchanged from the previous pipeline --
only `fit_property_direction` was dropped, since this port has no
property-direction traversal (PC traversal reuses `src/analysis/geometry.py`
instead).

Kept in its own module (separate from `layer_comparison.py`) since these
helpers are numeric/metric-only and don't touch data loading or orchestration.

Computation only -- never imports matplotlib.
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Dict, List

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import Descriptors

# -----------------------------------------------------------------------------
# Fixed property panel (same set used by 03_clustering's descriptor bars and
# the previous pipeline's PCA/semantics scripts, for direct comparability).
# -----------------------------------------------------------------------------


def _aromatic_fraction(mol) -> float:
    if mol.GetNumAtoms() == 0:
        return 0.0
    return sum(1 for a in mol.GetAtoms() if a.GetIsAromatic()) / mol.GetNumAtoms()


PROPERTY_FUNCS = {
    "MolWt": Descriptors.MolWt,
    "LogP": Descriptors.MolLogP,
    "TPSA": Descriptors.TPSA,
    "RingCount": Descriptors.RingCount,
    "Aromaticity": _aromatic_fraction,
}


def _property_worker(smi: str) -> List[float]:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return [np.nan] * len(PROPERTY_FUNCS)
    vals = []
    for fn in PROPERTY_FUNCS.values():
        try:
            vals.append(float(fn(mol)))
        except Exception:
            vals.append(np.nan)
    return vals


def compute_property_matrix(smiles_list: List[str], n_workers: int = 1) -> np.ndarray:
    """Parallel (via multiprocessing.Pool) RDKit property computation."""
    if n_workers <= 1:
        rows = [_property_worker(s) for s in tqdm(smiles_list, desc="Properties")]
    else:
        with mp.Pool(n_workers) as pool:
            rows = list(tqdm(pool.imap(_property_worker, smiles_list, chunksize=256),
                              total=len(smiles_list), desc=f"Properties ({n_workers} workers)"))
    return np.array(rows, dtype=np.float64)


# -----------------------------------------------------------------------------
# Cluster quality
# -----------------------------------------------------------------------------


def knn_purity(x_scaled: np.ndarray, labels: np.ndarray, k: int) -> float:
    k = min(k, len(x_scaled) - 1)
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=-1).fit(x_scaled)
    idx = nbrs.kneighbors(return_distance=False)[:, 1:]
    return float(np.mean(labels[idx] == labels[:, None]))


def compute_cluster_quality(x_scaled: np.ndarray, cluster_labels: np.ndarray,
                             k: int, include_silhouette: bool = False, seed: int = 1234) -> Dict[str, float]:
    out = {
        "calinski_harabasz": float(calinski_harabasz_score(x_scaled, cluster_labels)),
        "davies_bouldin": float(davies_bouldin_score(x_scaled, cluster_labels)),
        "knn_purity": knn_purity(x_scaled, cluster_labels, k),
    }
    if include_silhouette:
        out["silhouette"] = float(silhouette_score(
            x_scaled, cluster_labels, sample_size=min(10_000, len(x_scaled)), random_state=seed,
        ))
    return out


# -----------------------------------------------------------------------------
# Property linear-decodability
# -----------------------------------------------------------------------------


def compute_property_r2(x_scaled: np.ndarray, property_matrix: np.ndarray,
                         property_names: List[str], seed: int) -> Dict[str, float]:
    out = {}
    for p_idx, p_name in enumerate(property_names):
        y = property_matrix[:, p_idx]
        mask = np.isfinite(y)
        if mask.sum() < 20:
            out[p_name] = np.nan
            continue
        x_train, x_test, y_train, y_test = train_test_split(
            x_scaled[mask], y[mask], test_size=0.2, random_state=seed,
        )
        reg = LinearRegression().fit(x_train, y_train)
        out[p_name] = float(reg.score(x_test, y_test))
    return out


# -----------------------------------------------------------------------------
# ECFP-vs-embedding nearest-neighbor agreement
# -----------------------------------------------------------------------------


def stratified_subsample_indices(dataset_labels: np.ndarray, n_sample: int,
                                  rng: np.random.Generator) -> np.ndarray:
    dataset_labels = np.asarray(dataset_labels)
    unique_sources = sorted(set(dataset_labels.tolist()))
    n_total = len(dataset_labels)
    n_sample = min(n_sample, n_total)
    picked = []
    for src in unique_sources:
        src_idx = np.where(dataset_labels == src)[0]
        n_take = max(1, round(n_sample * len(src_idx) / n_total))
        n_take = min(n_take, len(src_idx))
        picked.append(rng.choice(src_idx, size=n_take, replace=False))
    idx = np.concatenate(picked)
    if len(idx) > n_sample:
        idx = rng.choice(idx, size=n_sample, replace=False)
    rng.shuffle(idx)
    return idx


def pairwise_tanimoto(ecfp_sub: np.ndarray, device: str) -> torch.Tensor:
    x = torch.from_numpy(ecfp_sub).to(device)
    row_sums = x.sum(dim=1)
    inter = x @ x.T
    union = row_sums.unsqueeze(1) + row_sums.unsqueeze(0) - inter
    return torch.where(union > 0, inter / union, torch.zeros_like(union))


def pairwise_cosine(embedding_sub: np.ndarray, device: str) -> torch.Tensor:
    x = torch.from_numpy(embedding_sub).to(device)
    x = torch.nn.functional.normalize(x, dim=1)
    return x @ x.T


def topk_overlap(sim_a: torch.Tensor, sim_b: torch.Tensor, k: int) -> float:
    n = sim_a.shape[0]
    k = min(k, n - 1)
    eye = torch.eye(n, dtype=torch.bool, device=sim_a.device)
    a = sim_a.masked_fill(eye, -float("inf"))
    b = sim_b.masked_fill(eye, -float("inf"))
    topk_a = a.topk(k, dim=1).indices
    topk_b = b.topk(k, dim=1).indices
    overlaps = [len(set(topk_a[i].tolist()) & set(topk_b[i].tolist())) / k for i in range(n)]
    return float(np.mean(overlaps))


def compute_nn_overlap(embedding_sub: np.ndarray, tanimoto: torch.Tensor,
                        k_values: List[int], device: str) -> Dict[str, float]:
    n = embedding_sub.shape[0]
    cosine = pairwise_cosine(embedding_sub, device)
    iu = torch.triu_indices(n, n, offset=1)
    tan_vals = tanimoto[iu[0], iu[1]].cpu().numpy()
    cos_vals = cosine[iu[0], iu[1]].cpu().numpy()
    r_pearson, _ = pearsonr(tan_vals, cos_vals)
    r_spearman, _ = spearmanr(tan_vals, cos_vals)
    out = {"tanimoto_cosine_pearson": float(r_pearson), "tanimoto_cosine_spearman": float(r_spearman)}
    for k in k_values:
        out[f"top{k}_overlap"] = topk_overlap(tanimoto, cosine, k)
    return out
