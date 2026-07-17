#!/usr/bin/env python
"""
property_directions_and_nn_overlap.py
========================================

Part 1 -- Property-direction traversals
-----------------------------------------
For a fixed panel of molecular properties (MolWt, LogP, TPSA, RingCount,
Aromaticity), fits a linear regression from the (scaled) embedding
(`global_cond` by default) to each property, then traverses directly along
that property's fitted regression direction -- exactly the same
molecule-image-strip + PC-space-position visual style as the PC traversal
script, except the traversal axis is "the direction that predicts MolWt"
etc. rather than a PCA axis.

Because the traversal direction `w_hat` is the *unit-normalized* regression
coefficient vector, projecting any embedding `e` onto it satisfies
`coef . e = ||coef|| * (e . w_hat)`, i.e. the fitted model's prediction is an
exact linear function of the 1D projection. This lets the bottom panel show
both the real scatter of (projection, actual property) *and* the exact
regression line, without approximation.

Part 2 -- ECFP vs. embedding nearest-neighbor overlap
--------------------------------------------------------
For a subsample of molecules (stratified across sources for the pooled
"cotrain" case, plain random for each individual source), computes:
    - pairwise ECFP Tanimoto similarity
    - pairwise `global_cond` cosine similarity
and reports:
    - top-k neighbor-set overlap (what fraction of a molecule's top-k ECFP
      neighbors are also top-k embedding neighbors), and
    - the Pearson/Spearman correlation between the two similarity measures
      across all pairs,
written to a single text file, plus a per-dataset (+ pooled) scatter of
embedding similarity vs. Tanimoto similarity in the same plot style used
throughout this pipeline (small, low-alpha, rasterized points).

Usage
-----
python 3_embedding_semantics.py --data-dir /scratch/gpfs/ZHONGE/jc4587/nmr_embs_FINAL --prefix cotrain --splits train --out-dir property_and_nn_summary
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
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Draw import rdMolDraw2D

# -----------------------------------------------------------------------------
# Property panel (Part 1)
# -----------------------------------------------------------------------------


def _aromatic_fraction(mol) -> float:
    atoms = mol.GetAtoms()
    if mol.GetNumAtoms() == 0:
        return 0.0
    return sum(1 for a in atoms if a.GetIsAromatic()) / mol.GetNumAtoms()


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


def compute_property_matrix(smiles_list: List[str], n_workers: int) -> pd.DataFrame:
    if n_workers <= 1:
        rows = [_property_worker(s) for s in tqdm(smiles_list, desc="Properties (MolWt/LogP/TPSA/Rings/Aromaticity)")]
    else:
        with mp.Pool(n_workers) as pool:
            rows = list(
                tqdm(pool.imap(_property_worker, smiles_list, chunksize=256),
                     total=len(smiles_list),
                     desc=f"Properties ({n_workers} workers)")
            )
    return pd.DataFrame(rows, columns=list(PROPERTY_FUNCS.keys()))


# -----------------------------------------------------------------------------
# Data loading (same convention as the rest of this pipeline)
# -----------------------------------------------------------------------------


def load_cotrain_data(data_dir: Path, prefix: str, splits: Sequence[str],
                       embedding_key: str) -> Dict:
    smiles: List[str] = []
    dataset: List[str] = []
    embed_parts: List[torch.Tensor] = []
    ecfp_parts: List[torch.Tensor] = []
    ecfp_radius = ecfp_nbits = None

    for split in splits:
        path = data_dir / f"{prefix}_{split}_global_cond.pt"
        if not path.exists():
            raise FileNotFoundError(f"Expected extraction output not found: {path}")
        print(f"Loading {path} ...")
        d = torch.load(path, map_location="cpu")

        if embedding_key not in d:
            raise KeyError(f"'{embedding_key}' not found in {path}. Keys: {list(d.keys())}")

        if ecfp_radius is None:
            ecfp_radius, ecfp_nbits = d["ecfp_radius"], d["ecfp_nbits"]

        smiles.extend(d["smiles"])
        dataset.extend(d["dataset"])
        embed_parts.append(d[embedding_key])
        ecfp_parts.append(d["ecfp"])

    embedding = torch.cat(embed_parts, dim=0)
    if embedding.dim() != 2:
        raise ValueError(
            f"Expected a 2D [N, D] tensor for '{embedding_key}', got shape "
            f"{tuple(embedding.shape)}."
        )
    ecfp = torch.cat(ecfp_parts, dim=0)

    return {
        "smiles": smiles,
        "dataset": np.asarray(dataset),
        "embedding": embedding.numpy().astype(np.float32),
        "ecfp": ecfp.numpy().astype(np.float32),
        "ecfp_radius": ecfp_radius,
        "ecfp_nbits": ecfp_nbits,
    }


# -----------------------------------------------------------------------------
# Part 1: property-direction regressions + traversal
# -----------------------------------------------------------------------------


def fit_property_direction(embedding_scaled: np.ndarray, property_values: np.ndarray,
                            seed: int) -> Tuple[np.ndarray, float, float]:
    """Returns (unit direction w_hat, coef_norm ||coef||, held-out test R^2)."""
    mask = np.isfinite(property_values)
    x = embedding_scaled[mask]
    y = property_values[mask]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=seed
    )

    reg = LinearRegression()
    reg.fit(x_train, y_train)
    r2 = reg.score(x_test, y_test)

    coef = reg.coef_
    coef_norm = float(np.linalg.norm(coef))
    w_hat = coef / (coef_norm + 1e-12)
    return w_hat, coef_norm, r2


def mol_to_image(smiles: str, size: int = 260) -> Optional[np.ndarray]:
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


def nearest_index_euclidean(query: np.ndarray, x: np.ndarray, device: str,
                             chunk_size: int = 100_000) -> int:
    q = torch.from_numpy(query).to(device)
    best_dist = None
    best_idx = -1
    n = x.shape[0]
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = torch.from_numpy(x[start:end]).to(device)
        dists = torch.linalg.norm(chunk - q, dim=1)
        m = dists.min()
        if best_dist is None or m < best_dist:
            best_dist = m
            best_idx = start + int(dists.argmin().item())
    return best_idx


def property_traversal_plot(prop_name: str, embedding_scaled: np.ndarray,
                             property_values: np.ndarray, smiles: List[str],
                             w_hat: np.ndarray, coef_norm: float, r2: float,
                             n_steps: int, device: str, out_path: Path,
                             mol_size: int = 260) -> None:
    mask = np.isfinite(property_values)
    x = embedding_scaled[mask]
    y = property_values[mask]
    smiles_masked = [s for s, m in zip(smiles, mask) if m]

    proj = x @ w_hat
    base = x.mean(axis=0)
    intercept = y.mean() - coef_norm * (base @ w_hat)  # exact along-direction fit

    proj_vals = np.linspace(proj.min(), proj.max(), n_steps)

    fig, axes = plt.subplots(2, n_steps, figsize=(n_steps * 2, 4))
    if n_steps == 1:
        axes = axes.reshape(2, 1)

    for i, val in enumerate(proj_vals):
        query = base + (val - base @ w_hat) * w_hat
        idx = nearest_index_euclidean(query, x, device)
        smi = smiles_masked[idx]
        img = mol_to_image(smi, size=mol_size)

        if img is not None:
            axes[0, i].imshow(img)
        axes[0, i].axis("off")
        axes[0, i].set_title(f"{prop_name}\u2248{intercept + coef_norm * val:.2f}", fontsize=6)

        axes[1, i].scatter(proj, y, s=0.3, alpha=0.2, c="lightgray", rasterized=True)
        line_x = np.array([proj.min(), proj.max()])
        axes[1, i].plot(line_x, intercept + coef_norm * line_x, c="steelblue", lw=1)
        axes[1, i].scatter(val, intercept + coef_norm * val, c="red", s=30, zorder=5)
        axes[1, i].set_xlabel("projection", fontsize=6)
        axes[1, i].set_ylabel(prop_name, fontsize=6)
        axes[1, i].tick_params(labelsize=5)
        for sp in ["top", "right"]:
            axes[1, i].spines[sp].set_visible(False)

    fig.suptitle(f"{prop_name} regression-direction traversal "
                 f"(held-out R\u00b2={r2:.3f})", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved {out_path}")


# -----------------------------------------------------------------------------
# Part 2: ECFP vs. embedding neighbor overlap
# -----------------------------------------------------------------------------


def stratified_subsample_indices(dataset_labels: np.ndarray, n_sample: int,
                                  rng: np.random.Generator) -> np.ndarray:
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
    sim = torch.where(union > 0, inter / union, torch.zeros_like(union))
    return sim


def pairwise_cosine(embedding_sub: np.ndarray, device: str) -> torch.Tensor:
    x = torch.from_numpy(embedding_sub).to(device)
    x = torch.nn.functional.normalize(x, dim=1)
    return x @ x.T


def topk_overlap(sim_a: torch.Tensor, sim_b: torch.Tensor, k: int) -> float:
    n = sim_a.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=sim_a.device)
    a = sim_a.masked_fill(eye, -float("inf"))
    b = sim_b.masked_fill(eye, -float("inf"))
    topk_a = a.topk(k, dim=1).indices
    topk_b = b.topk(k, dim=1).indices

    overlaps = []
    for i in range(n):
        set_a = set(topk_a[i].tolist())
        set_b = set(topk_b[i].tolist())
        overlaps.append(len(set_a & set_b) / k)
    return float(np.mean(overlaps))


def scatter_sim_vs_sim(tanimoto: torch.Tensor, cosine: torch.Tensor, label: str,
                        out_path: Path, max_points: int = 50_000,
                        seed: int = 1234) -> Tuple[float, float]:
    n = tanimoto.shape[0]
    iu = torch.triu_indices(n, n, offset=1)
    tan_vals = tanimoto[iu[0], iu[1]].cpu().numpy()
    cos_vals = cosine[iu[0], iu[1]].cpu().numpy()

    r_pearson, _ = pearsonr(tan_vals, cos_vals)
    r_spearman, _ = spearmanr(tan_vals, cos_vals)

    rng = np.random.default_rng(seed)
    if len(tan_vals) > max_points:
        plot_idx = rng.choice(len(tan_vals), size=max_points, replace=False)
    else:
        plot_idx = np.arange(len(tan_vals))

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(tan_vals[plot_idx], cos_vals[plot_idx], alpha=0.1, s=2,
               linewidths=0, rasterized=True)
    ax.set_title(f"{label}\nPearson r={r_pearson:.3f}, Spearman \u03c1={r_spearman:.3f}",
                 fontsize=10)
    ax.set_xlabel("ECFP Tanimoto similarity")
    ax.set_ylabel("global_cond cosine similarity")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved {out_path}")

    return r_pearson, r_spearman


def run_nn_overlap_analysis(label: str, embedding_sub: np.ndarray, ecfp_sub: np.ndarray,
                             k_values: List[int], device: str, out_dir: Path,
                             prefix: str, seed: int) -> Dict:
    tanimoto = pairwise_tanimoto(ecfp_sub, device)
    cosine = pairwise_cosine(embedding_sub, device)

    r_pearson, r_spearman = scatter_sim_vs_sim(
        tanimoto, cosine, label,
        out_dir / f"{prefix}_{label}_embedding_vs_tanimoto_scatter.png",
        seed=seed,
    )

    overlaps = {k: topk_overlap(tanimoto, cosine, k) for k in k_values}

    return {
        "label": label,
        "n_molecules": embedding_sub.shape[0],
        "pearson_r": r_pearson,
        "spearman_r": r_spearman,
        "topk_overlap": overlaps,
    }


def write_nn_overlap_report(results: List[Dict], out_path: Path) -> None:
    lines = [
        "ECFP (Tanimoto) vs. global_cond (cosine) nearest-neighbor overlap",
        "=" * 72,
        "",
        "For each group: Pearson/Spearman correlation between pairwise ECFP "
        "Tanimoto similarity and pairwise embedding cosine similarity, plus "
        "top-k neighbor-set overlap (fraction of a molecule's top-k ECFP "
        "neighbors that are also top-k embedding neighbors, averaged over "
        "all molecules in the sampled group).",
        "",
    ]
    for res in results:
        lines.append(f"--- {res['label']} (n={res['n_molecules']}) ---")
        lines.append(f"  Pearson r  (Tanimoto vs. cosine similarity): {res['pearson_r']:.4f}")
        lines.append(f"  Spearman rho                               : {res['spearman_r']:.4f}")
        for k, overlap in sorted(res["topk_overlap"].items()):
            lines.append(f"  Top-{k} neighbor-set overlap                : {overlap:.4f}")
        lines.append("")

    out_path.write_text("\n".join(lines))
    print(f"✓ Saved {out_path}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Property-direction traversals + ECFP/embedding "
                     "nearest-neighbor overlap analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--prefix", type=str, default="cotrain")
    p.add_argument("--splits", nargs="+", default=["train"],
                   choices=["train", "val", "test"])
    p.add_argument("--embedding-key", type=str, default="global_cond")
    p.add_argument("--out-dir", type=Path, default=Path("property_and_nn_summary"))

    # Part 1
    p.add_argument("--n-steps", type=int, default=8,
                   help="Steps per property traversal.")
    p.add_argument("--n-prop-workers", type=int, default=max(1, mp.cpu_count() - 2))

    # Part 2
    p.add_argument("--n-neighbor-sample", type=int, default=3000,
                   help="Subsample size for the NN-overlap analysis, per "
                        "dataset and for the pooled cotrain group.")
    p.add_argument("--k-neighbors", nargs="+", type=int, default=[5, 10, 25],
                   help="k values for top-k neighbor-set overlap.")

    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("Property-direction traversals + NN overlap analysis")
    print(f"  data dir : {args.data_dir}")
    print(f"  device   : {args.device}")
    print("=" * 78)

    data = load_cotrain_data(args.data_dir, args.prefix, args.splits, args.embedding_key)
    smiles = data["smiles"]
    embedding = data["embedding"]
    ecfp = data["ecfp"]
    dataset_labels = data["dataset"]
    print(f"Loaded {len(smiles)} molecules, embedding shape={embedding.shape}, "
          f"ecfp shape={ecfp.shape}")

    # =========================================================================
    # Part 1: property-direction regressions + traversals
    # =========================================================================

    scaler = StandardScaler()
    embedding_scaled = scaler.fit_transform(embedding)

    prop_df = compute_property_matrix(smiles, args.n_prop_workers)

    fit_rows = []
    for prop_name in PROPERTY_FUNCS:
        print(f"\n### Property direction: {prop_name} ###")
        property_values = prop_df[prop_name].to_numpy()
        w_hat, coef_norm, r2 = fit_property_direction(embedding_scaled, property_values, args.seed)
        print(f"  held-out R^2 = {r2:.4f}, ||coef|| = {coef_norm:.4f}")

        fit_rows.append({
            "property": prop_name, "r2_holdout": r2, "coef_norm": coef_norm,
        })

        property_traversal_plot(
            prop_name, embedding_scaled, property_values, smiles,
            w_hat, coef_norm, r2, args.n_steps, args.device,
            args.out_dir / f"{args.prefix}_property_traversal_{prop_name}.png",
        )

    fit_df = pd.DataFrame(fit_rows)
    fit_df.to_csv(args.out_dir / f"{args.prefix}_property_regression_fits.csv", index=False)
    print(f"\n✓ Saved {args.out_dir / f'{args.prefix}_property_regression_fits.csv'}")

    # =========================================================================
    # Part 2: ECFP vs. embedding nearest-neighbor overlap
    # =========================================================================

    print("\n" + "=" * 78)
    print("Nearest-neighbor overlap analysis")
    print("=" * 78)

    nn_results = []

    # Per-source-dataset groups
    for src in sorted(set(dataset_labels.tolist())):
        src_idx = np.where(dataset_labels == src)[0]
        n_take = min(args.n_neighbor_sample, len(src_idx))
        sub_idx = rng.choice(src_idx, size=n_take, replace=False)

        print(f"\n--- {src} (n={n_take}) ---")
        res = run_nn_overlap_analysis(
            src, embedding[sub_idx], ecfp[sub_idx], args.k_neighbors,
            args.device, args.out_dir, args.prefix, args.seed,
        )
        nn_results.append(res)

    # Pooled cotrain group (stratified across sources)
    cotrain_idx = stratified_subsample_indices(dataset_labels, args.n_neighbor_sample, rng)
    print(f"\n--- cotrain (pooled, n={len(cotrain_idx)}) ---")
    res = run_nn_overlap_analysis(
        "cotrain", embedding[cotrain_idx], ecfp[cotrain_idx], args.k_neighbors,
        args.device, args.out_dir, args.prefix, args.seed,
    )
    nn_results.append(res)

    write_nn_overlap_report(nn_results, args.out_dir / f"{args.prefix}_nn_overlap_stats.txt")

    print("\nDone.")


if __name__ == "__main__":
    main()
