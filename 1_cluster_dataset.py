#!/usr/bin/env python
"""
cluster_cotrain_molecules.py
=============================

Clusters the molecules extracted by `extract_cotrain_embeddings.py`
(`cotrain_{split}_global_cond.pt`) by chemical structure and produces:

    cotrain_cluster_labs.npy       -- int32 cluster label per molecule,
                                       aligned to the concatenated
                                       smiles/dataset order (see the
                                       companion `*_cluster_meta.csv`)
    cluster_representatives.png    -- same visual style as the earlier
                                       KMeans-based summary plot: a grid of
                                       cells, each showing representative
                                       molecule structures + a bar chart of
                                       normalized descriptor means.

Why not KMeans on raw ECFP bits (as the earlier script did)?
--------------------------------------------------------------
ECFPs are high-dimensional, sparse, binary vectors. KMeans assumes a
Euclidean space and represents each cluster by an arithmetic-mean
"centroid" -- for binary fingerprints that mean vector is not itself a
valid fingerprint and Euclidean distance does not track chemical
similarity well. The field-standard alternative is the **Taylor-Butina**
algorithm on **Tanimoto (Jaccard) distance**, which:
    - uses a metric designed for binary fingerprints (intersection/union),
    - doesn't require pre-specifying k -- only a similarity cutoff,
    - yields a genuine *exemplar molecule* per cluster (not an averaged,
      possibly-invalid fingerprint) as the natural "centroid",
    - is what RDKit's own clustering tooling implements
      (`rdkit.ML.Cluster.Butina`).

Butina's naive form is O(n^2), so it does not scale to the full cotrain
corpus (potentially >1M molecules). This script uses the standard
scalable pattern:
    1. Take a stratified subsample (stratified by source dataset) of size
       `--n-cluster-sample`.
    2. Run exact Butina clustering on that subsample using a vectorized
       Tanimoto distance computation.
    3. Treat each cluster's Butina-designated exemplar (its centroid
       molecule) as a prototype fingerprint.
    4. Assign every remaining molecule in the full dataset to its nearest
       prototype by Tanimoto similarity (chunked matrix multiplication,
       optionally GPU-accelerated via torch) -- this is O(N x k), not
       O(N^2), and stays linear in the full dataset size.

Usage
-----
python 1_cluster_dataset.py --data-dir /scratch/gpfs/ZHONGE/jc4587/nmr_embs_FINAL --prefix cotrain --splits train --out-dir cluster_summary --n-cluster-sample 10000 --butina-cutoff 0.35
"""

from __future__ import annotations

import argparse
import io
import json
import math
import multiprocessing as mp
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import Draw, Descriptors
from rdkit.Chem import rdCoordGen

# -----------------------------------------------------------------------------
# Molecule drawing (per user's snippet: rdCoordGen for nicer 2D layout, SVG
# render). Falls back to plain RDKit raster grid images if an SVG->raster
# backend (cairosvg) isn't installed, so the script never hard-fails on a
# missing optional dependency.
# -----------------------------------------------------------------------------


def smiles_to_svg(smiles: str, svg_path: Optional[str] = None,
                   image_size: Tuple[int, int] = (300, 300)) -> Optional[str]:
    """
    Convert a SMILES string to an SVG representation of the molecule, using
    rdCoordGen for nicer 2D coordinates.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Warning: Could not create molecule from SMILES: {smiles}")
        return None

    try:
        rdCoordGen.AddCoords(mol)
    except Exception as coord_e:
        print(f"Warning: rdCoordGen failed: {coord_e}. Falling back to default coordinates.")

    drawer = Draw.rdMolDraw2D.MolDraw2DSVG(image_size[0], image_size[1])
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg_data = drawer.GetDrawingText()

    if svg_path:
        with open(svg_path, "w") as f:
            f.write(svg_data)

    return svg_data


def _svg_to_pil(svg_data: str, size: Tuple[int, int]) -> Optional[Image.Image]:
    """Best-effort SVG -> PIL.Image conversion. Returns None if no SVG
    rasterizer is available so callers can fall back gracefully."""
    try:
        import cairosvg  # optional dependency
        png_bytes = cairosvg.svg2png(
            bytestring=svg_data.encode("utf-8"),
            output_width=size[0], output_height=size[1],
        )
        return Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    except Exception:
        return None


def render_mol_grid(smiles_list: List[str], mols_per_row: int = 3,
                     sub_img_size: Tuple[int, int] = (180, 180)) -> Image.Image:
    """Render a grid of representative molecules as one PIL image.

    Tries the rdCoordGen + SVG path per molecule first (nicer, more uniform
    2D layouts). If SVG rasterization isn't available in this environment,
    falls back to RDKit's built-in raster grid image for the whole set."""
    tiles = []
    all_ok = True
    for smi in smiles_list:
        svg = smiles_to_svg(smi, image_size=sub_img_size)
        pil_img = _svg_to_pil(svg, sub_img_size) if svg is not None else None
        if pil_img is None:
            all_ok = False
            break
        tiles.append(pil_img)

    if all_ok and tiles:
        n = len(tiles)
        ncols = mols_per_row
        nrows = math.ceil(n / ncols)
        w, h = sub_img_size
        canvas = Image.new("RGBA", (ncols * w, nrows * h), (255, 255, 255, 255))
        for i, tile in enumerate(tiles):
            r, c = divmod(i, ncols)
            canvas.paste(tile, (c * w, r * h), tile)
        return canvas

    # Fallback: plain RDKit raster grid (no cairosvg available).
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    mols = [m for m in mols if m is not None]
    img = Draw.MolsToGridImage(
        mols, molsPerRow=mols_per_row, subImgSize=sub_img_size, returnPNG=False,
    )
    return img.convert("RGBA") if hasattr(img, "convert") else img


# -----------------------------------------------------------------------------
# Data loading
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
        print(f"Loading {path} ...")
        d = torch.load(path, map_location="cpu")

        if ecfp_radius is None:
            ecfp_radius, ecfp_nbits = d["ecfp_radius"], d["ecfp_nbits"]
        else:
            assert (ecfp_radius, ecfp_nbits) == (d["ecfp_radius"], d["ecfp_nbits"]), (
                "Mismatched ECFP radius/nbits across splits -- "
                "was the extraction script run with consistent settings?"
            )

        n = len(d["smiles"])
        smiles.extend(d["smiles"])
        dataset.extend(d["dataset"])
        split_tag.extend([split] * n)
        ecfp_parts.append(d["ecfp"])

    ecfp = torch.cat(ecfp_parts, dim=0).numpy().astype(np.float32)

    return {
        "smiles": smiles,
        "dataset": dataset,
        "split": split_tag,
        "ecfp": ecfp,
        "ecfp_radius": ecfp_radius,
        "ecfp_nbits": ecfp_nbits,
    }


# -----------------------------------------------------------------------------
# Stratified subsampling (by source dataset)
# -----------------------------------------------------------------------------


def stratified_subsample_indices(dataset_labels: List[str], n_sample: int,
                                  rng: np.random.Generator) -> np.ndarray:
    dataset_labels = np.asarray(dataset_labels)
    unique_sources = sorted(set(dataset_labels.tolist()))
    n_sources = len(unique_sources)
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


# -----------------------------------------------------------------------------
# Tanimoto distance + Butina clustering on the subsample
# -----------------------------------------------------------------------------


def tanimoto_condensed_distances(ecfp_sub: np.ndarray, device: str) -> List[float]:
    """Returns the condensed distance list in the exact order RDKit's
    Butina.ClusterData expects when isDistData=True:
        for i in range(1, n): for j in range(i): dist(i, j)
    computed via vectorized Tanimoto (intersection/union) on 0/1 rows.
    """
    x = torch.from_numpy(ecfp_sub).to(device)
    row_sums = x.sum(dim=1)  # popcount per fingerprint

    dists: List[float] = []
    n = x.shape[0]
    for i in tqdm(range(1, n), desc="Pairwise Tanimoto (subsample)"):
        inter = (x[i : i + 1] @ x[:i].T).squeeze(0)  # [i]
        union = row_sums[i] + row_sums[:i] - inter
        sim = torch.where(union > 0, inter / union, torch.zeros_like(union))
        dists.extend((1.0 - sim).cpu().tolist())
    return dists


def run_butina_clustering(ecfp_sub: np.ndarray, cutoff: float, device: str):
    from rdkit.ML.Cluster import Butina

    n = ecfp_sub.shape[0]
    dists = tanimoto_condensed_distances(ecfp_sub, device)
    clusters = Butina.ClusterData(dists, n, cutoff, isDistData=True)
    # clusters: tuple of tuples of subsample-local indices, largest first,
    # each cluster's index 0 is its Butina-designated centroid/exemplar.
    return clusters


# -----------------------------------------------------------------------------
# Nearest-prototype assignment for the full dataset (scalable, O(N x k))
# -----------------------------------------------------------------------------


def assign_to_prototypes(ecfp_all: np.ndarray, prototype_fps: np.ndarray,
                          device: str, chunk_size: int = 50_000) -> np.ndarray:
    protos = torch.from_numpy(prototype_fps).to(device)          # [k, d]
    proto_sums = protos.sum(dim=1)                                # [k]

    n = ecfp_all.shape[0]
    labels = np.empty(n, dtype=np.int32)

    for start in tqdm(range(0, n, chunk_size), desc="Assigning molecules to clusters"):
        end = min(start + chunk_size, n)
        x = torch.from_numpy(ecfp_all[start:end]).to(device)      # [b, d]
        x_sums = x.sum(dim=1, keepdim=True)                       # [b, 1]

        inter = x @ protos.T                                      # [b, k]
        union = x_sums + proto_sums.unsqueeze(0) - inter          # [b, k]
        sim = torch.where(union > 0, inter / union, torch.zeros_like(union))

        labels[start:end] = sim.argmax(dim=1).cpu().numpy().astype(np.int32)

    return labels


# -----------------------------------------------------------------------------
# RDKit descriptors (parallel)
# -----------------------------------------------------------------------------


def _descriptor_worker(smi: str) -> Optional[Dict[str, float]]:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "Rings": Descriptors.RingCount(mol),
    }


def compute_descriptors(smiles_list: List[str], n_workers: int) -> pd.DataFrame:
    if n_workers <= 1:
        rows = [_descriptor_worker(s) for s in tqdm(smiles_list, desc="RDKit descriptors")]
    else:
        with mp.Pool(n_workers) as pool:
            rows = list(
                tqdm(pool.imap(_descriptor_worker, smiles_list, chunksize=256),
                     total=len(smiles_list), desc=f"RDKit descriptors ({n_workers} workers)")
            )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Plotting (same layout/style as the earlier KMeans summary script)
# -----------------------------------------------------------------------------

PROPS = ["MolWt", "LogP", "TPSA", "Rings"]


def plot_cluster_representatives(stats_df: pd.DataFrame, cluster_reps: Dict[int, List[str]],
                                  out_path: Path, mols_per_row: int = 3,
                                  sub_img_size: Tuple[int, int] = (180, 180)) -> None:
    n = len(stats_df)
    ncols = 5
    nrows = math.ceil(n / ncols)

    fig = plt.figure(figsize=(14, 4 * nrows))
    outer = fig.add_gridspec(nrows, ncols, wspace=0.1, hspace=0.4)

    X = stats_df[PROPS].copy()
    X = (X - X.min()) / (X.max() - X.min()).replace(0, 1)

    for k, ((_, row), (_, vals)) in enumerate(zip(stats_df.iterrows(), X.iterrows())):
        r, c = divmod(k, ncols)
        gs = outer[r, c].subgridspec(2, 1, height_ratios=[2.5, 1], hspace=0.02)
        ax_img = fig.add_subplot(gs[0])
        ax_bar = fig.add_subplot(gs[1])

        cluster = int(row.cluster)
        rep_smiles = cluster_reps[cluster]

        img = render_mol_grid(rep_smiles, mols_per_row=mols_per_row, sub_img_size=sub_img_size)
        ax_img.imshow(np.asarray(img))
        ax_img.axis("off")
        ax_img.set_title(f"Cluster {cluster} (n={int(row.n_molecules)})", fontsize=10)

        ax_bar.bar(range(len(PROPS)), vals.values)
        ax_bar.set_ylim(0, 1)
        ax_bar.set_xticks(range(len(PROPS)))
        ax_bar.set_xticklabels(PROPS, rotation=45, fontsize=8)
        ax_bar.set_yticks([])

    for k in range(n, nrows * ncols):
        r, c = divmod(k, ncols)
        gs = outer[r, c].subgridspec(2, 1)
        fig.add_subplot(gs[0]).axis("off")
        fig.add_subplot(gs[1]).axis("off")

    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cluster cotrain molecules by ECFP (Butina/Tanimoto) and "
                     "plot cluster representatives.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", type=Path, required=True,
                   help="Directory containing <prefix>_<split>_global_cond.pt "
                        "(the output of extract_cotrain_embeddings.py).")
    p.add_argument("--prefix", type=str, default="cotrain")
    p.add_argument("--splits", nargs="+", default=["train"],
                   choices=["train", "val", "test"],
                   help="Which extracted splits to pool together for clustering.")
    p.add_argument("--out-dir", type=Path, default=Path("cluster_summary"))
    p.add_argument("--n-cluster-sample", type=int, default=10_000,
                   help="Size of the stratified subsample used for exact "
                        "Butina clustering (O(n^2) step -- keep this in the "
                        "low tens of thousands).")
    p.add_argument("--butina-cutoff", type=float, default=0.35,
                   help="Tanimoto DISTANCE cutoff for Butina clustering "
                        "(i.e. similarity >= 1 - cutoff joins a cluster). "
                        "0.3-0.4 is typical for ECFP4/Morgan r=2.")
    p.add_argument("--max-clusters-plot", type=int, default=24,
                   help="Only the top-K clusters by size are drawn in "
                        "cluster_representatives.png (all clusters still "
                        "get labels in cotrain_cluster_labs.npy).")
    p.add_argument("--n-reps", type=int, default=6,
                   help="Representative molecules drawn per cluster.")
    p.add_argument("--assign-chunk-size", type=int, default=50_000)
    p.add_argument("--n-desc-workers", type=int, default=max(1, mp.cpu_count() - 2))
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args(argv)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("Cotrain molecule clustering (Butina / Tanimoto)")
    print(f"  data dir          : {args.data_dir}")
    print(f"  splits            : {args.splits}")
    print(f"  cluster subsample : {args.n_cluster_sample}")
    print(f"  butina cutoff     : {args.butina_cutoff}")
    print(f"  device            : {args.device}")
    print("=" * 78)

    data = load_cotrain_data(args.data_dir, args.prefix, args.splits)
    smiles = data["smiles"]
    dataset_labels = data["dataset"]
    ecfp = data["ecfp"]
    n_total = len(smiles)
    print(f"Loaded {n_total} molecules "
          f"(ECFP: radius={data['ecfp_radius']}, nBits={data['ecfp_nbits']})")

    # ---- 1. Stratified subsample for exact Butina clustering ----
    sub_idx = stratified_subsample_indices(dataset_labels, args.n_cluster_sample, rng)
    ecfp_sub = ecfp[sub_idx]
    print(f"Subsampled {len(sub_idx)} molecules for exact clustering "
          f"(stratified by source dataset).")

    # ---- 2. Butina clustering on the subsample ----
    clusters = run_butina_clustering(ecfp_sub, args.butina_cutoff, args.device)
    print(f"Butina produced {len(clusters)} clusters on the subsample "
          f"(cutoff={args.butina_cutoff}).")

    # Map subsample-local cluster indices back to global molecule indices,
    # and collect each cluster's centroid (exemplar) fingerprint.
    sub_labels_local = np.full(len(sub_idx), -1, dtype=np.int32)
    prototype_fps = []
    for cluster_id, member_local_idx in enumerate(clusters):
        for li in member_local_idx:
            sub_labels_local[li] = cluster_id
        centroid_local = member_local_idx[0]
        prototype_fps.append(ecfp_sub[centroid_local])
    prototype_fps = np.stack(prototype_fps, axis=0)

    # ---- 3. Assign every molecule in the full dataset to nearest prototype ----
    labels = assign_to_prototypes(ecfp, prototype_fps, args.device, args.assign_chunk_size)

    # Overwrite subsample members with their exact Butina assignment (rather
    # than the nearest-prototype approximation) so the exemplar set's own
    # clustering stays exact.
    labels[sub_idx] = sub_labels_local

    np.save(args.out_dir / "cotrain_cluster_labs.npy", labels)
    print(f"✓ Saved {args.out_dir / 'cotrain_cluster_labs.npy'} "
          f"(shape={labels.shape}, n_clusters={len(clusters)})")

    # Alignment companion file -- not one of the two requested outputs, but
    # cheap and otherwise the label array has no way to be joined back to
    # smiles/source/split.
    meta_df = pd.DataFrame({
        "smiles": smiles, "dataset": dataset_labels, "split": data["split"],
        "cluster": labels,
    })
    meta_df.to_csv(args.out_dir / f"{args.prefix}_cluster_meta.csv", index=False)

    # ---- 4. Descriptors + per-cluster summary ----
    desc_df = compute_descriptors(smiles, args.n_desc_workers)

    cluster_stats = []
    cluster_reps: Dict[int, List[str]] = {}
    unique_clusters, counts = np.unique(labels, return_counts=True)
    order = np.argsort(-counts)  # largest first

    for cluster in unique_clusters[order]:
        idx = np.where(labels == cluster)[0]
        means = desc_df.iloc[idx].mean(numeric_only=True)
        row = {"cluster": int(cluster), "n_molecules": len(idx)}
        row.update(means.to_dict())
        cluster_stats.append(row)

    stats_df = pd.DataFrame(cluster_stats)
    stats_df.to_csv(args.out_dir / "cluster_stats.csv", index=False)
    print(f"✓ Saved {args.out_dir / 'cluster_stats.csv'}")

    # Representatives: the Butina exemplar (if this cluster came from the
    # subsample and still exists) plus a random fill from the rest of the
    # cluster's members, for chemical diversity in the plotted grid.
    plot_stats_df = stats_df.head(args.max_clusters_plot).reset_index(drop=True)
    for cluster in plot_stats_df["cluster"]:
        idx = np.where(labels == int(cluster))[0]
        n_reps = min(args.n_reps, len(idx))
        rep_idx = rng.choice(idx, size=n_reps, replace=False)
        cluster_reps[int(cluster)] = [smiles[i] for i in rep_idx]

    plot_cluster_representatives(
        plot_stats_df, cluster_reps, args.out_dir / "cluster_representatives.png",
    )
    print(f"✓ Saved {args.out_dir / 'cluster_representatives.png'} "
          f"(top {len(plot_stats_df)} of {len(stats_df)} clusters)")

    manifest = {
        "data_dir": str(args.data_dir), "prefix": args.prefix, "splits": args.splits,
        "n_total_molecules": n_total, "n_cluster_sample": args.n_cluster_sample,
        "butina_cutoff": args.butina_cutoff, "n_clusters": int(len(clusters)),
        "ecfp_radius": data["ecfp_radius"], "ecfp_nbits": data["ecfp_nbits"],
        "seed": args.seed,
    }
    (args.out_dir / f"{args.prefix}_clustering_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    print("Done.")


if __name__ == "__main__":
    main()
