#!/usr/bin/env python
"""
umap_main_figure.py
======================

Generalized rewrite of the original "2a_pretty_plot" UMAP figure script.
Preserves its exact visual style (rainbow cluster colors + gray "other",
thin zoom rectangles with matching-color cluster-id labels, side-by-side
scatter-zoom + molecule-grid inset panels) while adding:

    1. Dataset subsetting -- plot only e.g. {spectranp} or {spectranp,
       nmrexp} instead of always all three cotrain sources.
    2. SMILES-of-interest highlighting -- give a list of SMILES strings and
       the script finds each in the loaded data (via canonical-SMILES
       match), marks it with a star on the main scatter, and saves a
       zoom + molecule-grid inset centered on it (neighborhood molecules,
       with the query molecule labeled "query" in its grid legend).
    3. Cluster highlighting is now fully optional -- clusters are only
       colored/gray-masked/inset if you pass `--highlight-clusters`,
       instead of the original's hardcoded `clusters_to_show`.
    4. NEW: --color-by np_class -- color by NPClassifier's dominant
       natural-product class per molecule (top-N most common classes,
       everything else bucketed into a gray "Other").
    5. NEW: --color-by sim_real -- color by whether each molecule's
       `dataset` source is real vs. synthetic, via configurable prefix
       matching (--real-prefixes / --sim-prefixes), with an "unknown"
       bucket for anything that matches neither.

Background coloring of the full scatter (`--color-by`) is independent of
the highlight overlays (clusters / SMILES-of-interest), so you can e.g.
color by NP class while still drawing cluster-zoom insets on top.

Inputs
------
    - `<prefix>_<split>_global_cond.pt`  (extract_cotrain_embeddings.py)
    - `cotrain_cluster_labs.npy`          (cluster_cotrain_molecules.py;
                                            only needed for --color-by
                                            cluster or --highlight-clusters)
    - `labels.pkl`                        (npclassifier_local.py; only
                                            needed for --color-by np_class)
    - a precomputed 2D projection (--umap-embedding-path), OR the script
      fits UMAP itself on the (possibly subsetted) embeddings and can cache
      the result via --save-umap-embedding.

Usage
-----
    # Reproduce the original figure: all 3 sources, 5 highlighted clusters
    python umap_main_figure.py \\
        --data-dir /scratch/gpfs/ZHONGE/jc4587/nmr_embs_cotrain \\
        --cluster-labels-path cluster_summary/cotrain_cluster_labs.npy \\
        --color-by cluster --highlight-clusters 0 48 99 149 199 \\
        --out-dir umap_figure

    # Color by NPClassifier's top-10 most common classes
    python umap_main_figure.py \\
        --data-dir /scratch/gpfs/ZHONGE/jc4587/nmr_embs_cotrain \\
        --np-class-labels-path labels.pkl --color-by np_class \\
        --np-class-top-n 10 --out-dir umap_figure_npclass

    # Color by real vs. synthetic source
    python umap_main_figure.py \\
        --data-dir /scratch/gpfs/ZHONGE/jc4587/nmr_embs_cotrain \\
        --color-by sim_real --out-dir umap_figure_simreal
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

QUALITATIVE_PALETTE = plt.cm.tab10(np.linspace(0, 1, 10))
GRAY = np.array([0.75, 0.75, 0.75, 1.0])

WINDOW_SIZE_GLOBAL = 2.5  # set from --window-size in main(); module-level for plot fns


# -----------------------------------------------------------------------------
# Data loading
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
            raise KeyError(f"'{embedding_key}' not found in {path}. Keys: {list(d.keys())}")
        smiles.extend(d["smiles"])
        dataset.extend(d["dataset"])
        embed_parts.append(d[embedding_key])

    embedding = torch.cat(embed_parts, dim=0).numpy().astype(np.float32)
    return {"smiles": smiles, "dataset": np.asarray(dataset), "embedding": embedding}


def load_pickle_labels(path: Path, expected_len: int, label_name: str) -> np.ndarray:
    """Loads a pickled List[str]/List[float] (e.g. labels.pkl from
    npclassifier_local.py) and checks it's aligned to the unfiltered data."""
    with open(path, "rb") as f:
        labels = pickle.load(f)
    labels = np.asarray(labels, dtype=object)
    if len(labels) != expected_len:
        raise ValueError(
            f"{path} has {len(labels)} {label_name} entries but the loaded "
            f"(unfiltered) data has {expected_len} molecules -- check that "
            f"--splits matches whatever split(s) {label_name} was generated for "
            f"(e.g. npclassifier_local.py labels.pkl is typically train-only)."
        )
    return labels


def filter_by_datasets(data: Dict, datasets: Optional[Sequence[str]],
                        extra_label_arrays: Dict[str, Optional[np.ndarray]]
                        ) -> Tuple[Dict, Dict[str, Optional[np.ndarray]]]:
    if datasets is None:
        return data, extra_label_arrays

    mask = np.isin(data["dataset"], list(datasets))
    n_before, n_after = len(data["dataset"]), int(mask.sum())
    if n_after == 0:
        raise ValueError(f"No molecules found for --datasets {datasets}. "
                          f"Available sources: {sorted(set(data['dataset'].tolist()))}")
    print(f"Subsetting to datasets={list(datasets)}: {n_before} -> {n_after} molecules")

    filtered = {
        "smiles": [s for s, m in zip(data["smiles"], mask) if m],
        "dataset": data["dataset"][mask],
        "embedding": data["embedding"][mask],
    }
    filtered_extras = {
        name: (arr[mask] if arr is not None else None)
        for name, arr in extra_label_arrays.items()
    }
    return filtered, filtered_extras


# -----------------------------------------------------------------------------
# UMAP (load-or-compute, cacheable)
# -----------------------------------------------------------------------------

def reduce_with_pca(embedding: np.ndarray, n_components: Optional[int], seed: int) -> np.ndarray:
    """Reduces to n_components via PCA before UMAP. Skipped if n_components
    is None/0, or if the embedding already has <= n_components dimensions
    (PCA can't add dimensions, and there's nothing to gain)."""
    if not n_components:
        return embedding
    if embedding.shape[1] <= n_components:
        print(f"Skipping PCA: embedding already has {embedding.shape[1]} dims "
              f"(<= requested {n_components}).")
        return embedding

    print(f"Reducing {embedding.shape[1]} dims -> {n_components} dims via PCA "
          f"before UMAP (on {embedding.shape[0]} points) ...")
    pca = PCA(n_components=n_components, random_state=seed)
    reduced = pca.fit_transform(embedding).astype(np.float32)
    explained = pca.explained_variance_ratio_.sum()
    print(f"PCA done. {n_components} components explain "
          f"{explained * 100:.1f}% of variance.")
    return reduced


def load_or_compute_umap(embedding: np.ndarray, load_path: Optional[Path],
                          save_path: Optional[Path], n_neighbors: int,
                          min_dist: float, metric: str, seed: int,
                          pca_dim: Optional[int] = 50) -> np.ndarray: 
    if load_path is not None:
        print(f"Loading precomputed 2D projection from {load_path}")
        emb2d = np.load(load_path)
        if emb2d.shape[0] != embedding.shape[0]:
            raise ValueError(
                f"{load_path} has {emb2d.shape[0]} rows but the (possibly "
                f"subsetted) loaded data has {embedding.shape[0]} molecules "
                f"-- a precomputed projection must be aligned to the exact "
                f"subset being plotted."
            )
        return emb2d

    import umap.umap_ as umap_lib

    embedding = reduce_with_pca(embedding, pca_dim, seed)
    if pca_dim and embedding.shape[1] == pca_dim and metric not in ("euclidean", "cosine"):
        print(f"[warn] metric='{metric}' was requested for UMAP, but the input "
              f"to UMAP is now PCA-reduced (dense, continuous) rather than the "
              f"original representation. Binary-fingerprint-oriented metrics "
              f"like 'jaccard' don't really apply to PCA output -- consider "
              f"--umap-metric euclidean or cosine instead, or --pca-dim 0 to "
              f"disable PCA and keep the original metric meaningful.")

    print(f"Fitting UMAP (n_neighbors={n_neighbors}, min_dist={min_dist}, "
          f"metric={metric}) on {embedding.shape[0]} points ...")
    reducer = umap_lib.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                             metric=metric, random_state=seed, n_jobs=-1)
    emb2d = reducer.fit_transform(embedding)

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, emb2d)
        print(f"Cached 2D projection to {save_path}")

    return emb2d


# -----------------------------------------------------------------------------
# SMILES-of-interest matching
# -----------------------------------------------------------------------------

def canonicalize(smi: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def match_smiles_of_interest(query_smiles: Sequence[str],
                              dataset_smiles: List[str]) -> Dict[str, int]:
    """Returns {query_smiles: index_in_dataset}, skipping (with a warning)
    any query that doesn't canonicalize or isn't found in the dataset."""
    canon_to_idx: Dict[str, int] = {}
    for i, s in enumerate(dataset_smiles):
        c = canonicalize(s)
        if c is not None and c not in canon_to_idx:
            canon_to_idx[c] = i

    matches = {}
    for q in query_smiles:
        cq = canonicalize(q)
        if cq is None:
            print(f"[warn] Could not parse query SMILES, skipping: {q!r}")
            continue
        if cq not in canon_to_idx:
            print(f"[warn] Query SMILES not found in the loaded/subsetted "
                  f"data, skipping: {q!r}")
            continue
        matches[q] = canon_to_idx[cq]
    return matches


# -----------------------------------------------------------------------------
# Region definitions (clusters and/or SMILES-of-interest)
# -----------------------------------------------------------------------------

class Region:
    def __init__(self, kind: str, region_id, label: str, color, center: Tuple[float, float],
                 member_idx: np.ndarray, query_idx: Optional[int] = None):
        self.kind = kind                # "cluster" | "smiles"
        self.region_id = region_id      # cluster int id, or the raw query smiles string
        self.label = label              # text drawn on the main plot
        self.color = color
        self.center = center
        self.member_idx = member_idx    # indices used to sample the molecule-grid panel
        self.query_idx = query_idx      # for "smiles" regions: the exact matched index


def build_cluster_regions(cluster_ids: Sequence[int], labels: np.ndarray,
                           emb2d: np.ndarray) -> List[Region]:
    n_normal = int(labels.max()) + 1
    base_colors = plt.cm.rainbow(np.linspace(0, 1, n_normal))

    regions = []
    for cid in cluster_ids:
        cluster_mask = labels == cid
        pts = emb2d[cluster_mask]
        if len(pts) == 0:
            print(f"[warn] Cluster {cid} has no members in the current "
                  f"(possibly subsetted) data, skipping.")
            continue
        center = (float(np.median(pts[:, 0])), float(np.median(pts[:, 1])))
        member_idx = np.where(cluster_mask)[0]
        color = base_colors[cid % n_normal]
        regions.append(Region("cluster", cid, str(cid), color, center, member_idx))
    return regions


def build_smiles_regions(matches: Dict[str, int], emb2d: np.ndarray,
                          window_size: float) -> List[Region]:
    regions = []
    for i, (query, idx) in enumerate(matches.items()):
        center = (float(emb2d[idx, 0]), float(emb2d[idx, 1]))
        half = window_size / 2
        in_window = (
            (emb2d[:, 0] >= center[0] - half) & (emb2d[:, 0] <= center[0] + half) &
            (emb2d[:, 1] >= center[1] - half) & (emb2d[:, 1] <= center[1] + half)
        )
        member_idx = np.where(in_window)[0]
        color = QUALITATIVE_PALETTE[i % len(QUALITATIVE_PALETTE)]
        label = f"S{i + 1}"
        regions.append(Region("smiles", query, label, color, center, member_idx, query_idx=idx))
    return regions


# -----------------------------------------------------------------------------
# Molecule sampling + rendering (same primitives as the original script)
# -----------------------------------------------------------------------------

def sample_region_smiles(region: Region, smiles: List[str], n_mols: int,
                          seed: int) -> Tuple[List, List[str]]:
    """Returns (rdkit_mols, legends). For 'smiles' regions the query molecule
    is always included first and legended 'query'."""
    rng = random.Random(seed + (region.query_idx if region.query_idx is not None
                                 else hash(region.region_id) % 10_000))

    ordered_idx = list(region.member_idx)
    rng.shuffle(ordered_idx)

    chosen_idx = []
    if region.kind == "smiles" and region.query_idx is not None:
        chosen_idx.append(region.query_idx)

    for idx in ordered_idx:
        if len(chosen_idx) >= n_mols:
            break
        if idx in chosen_idx:
            continue
        if Chem.MolFromSmiles(smiles[idx]) is None:
            continue
        chosen_idx.append(idx)

    mols, legends = [], []
    for idx in chosen_idx:
        mol = Chem.MolFromSmiles(smiles[idx])
        if mol is None:
            continue
        mols.append(mol)
        legends.append("query" if (region.kind == "smiles" and idx == region.query_idx) else "")
    return mols, legends


def mols_to_image(mols: List, legends: Optional[List[str]] = None,
                   mols_per_row: int = 2, sub_img_size: Tuple[int, int] = (180, 180)):
    if len(mols) == 0:
        return None
    return Draw.MolsToGridImage(
        mols, molsPerRow=mols_per_row, subImgSize=sub_img_size,
        legends=legends, returnPNG=False,
    )


# -----------------------------------------------------------------------------
# Background coloring for the full scatter
# -----------------------------------------------------------------------------

def color_by_dataset(dataset_labels: np.ndarray) -> Tuple[np.ndarray, Dict[str, tuple]]:
    sources = sorted(set(dataset_labels.tolist()))
    palette = {src: QUALITATIVE_PALETTE[i % len(QUALITATIVE_PALETTE)] for i, src in enumerate(sources)}
    colors = np.array([palette[s] for s in dataset_labels])
    return colors, palette


def color_by_cluster(labels: np.ndarray, highlight_clusters: Optional[Sequence[int]]
                      ) -> Tuple[np.ndarray, ListedColormap, BoundaryNorm, np.ndarray]:
    n_normal = int(labels.max()) + 1
    base_colors = plt.cm.rainbow(np.linspace(0, 1, n_normal))
    gray = np.array([[0.75, 0.75, 0.75, 1.0]])
    custom_cmap = ListedColormap(np.vstack([base_colors, gray]))

    labels_plot = labels.copy()
    special_label = n_normal  # == labels.max() + 1

    if highlight_clusters is not None:
        mask = ~np.isin(labels_plot, list(highlight_clusters))
        labels_plot[mask] = special_label

    bounds = np.arange(special_label + 2) - 0.5
    norm = BoundaryNorm(bounds, custom_cmap.N)
    return labels_plot, custom_cmap, norm, base_colors


def _bucket_by_frequency(labels: np.ndarray, top_n: int, other_label: str = "Other"
                          ) -> np.ndarray:
    """Keeps the top_n most frequent distinct values as-is; everything else
    becomes `other_label`."""
    vals, counts = np.unique(labels, return_counts=True)
    order = np.argsort(-counts)
    keep = set(vals[order][:top_n].tolist())
    return np.array([v if v in keep else other_label for v in labels], dtype=object)


def color_by_np_class(np_class_labels: np.ndarray, top_n: int
                       ) -> Tuple[np.ndarray, Dict[str, tuple]]:
    """Colors by NPClassifier's dominant class per molecule, keeping only the
    top_n most common classes distinctly colored and bucketing everything
    else (long-tail classes) into a gray 'Other'."""
    bucketed = _bucket_by_frequency(np_class_labels, top_n, other_label="Other")
    categories = sorted(c for c in set(bucketed.tolist()) if c != "Other")

    palette = {cat: QUALITATIVE_PALETTE[i % len(QUALITATIVE_PALETTE)] for i, cat in enumerate(categories)}
    if "Other" in bucketed:
        palette["Other"] = tuple(GRAY)

    colors = np.array([palette[c] for c in bucketed])
    return colors, palette


def classify_sim_real(dataset_labels: np.ndarray, real_prefixes: Sequence[str],
                       sim_prefixes: Sequence[str]) -> np.ndarray:
    real_prefixes = [p.lower() for p in real_prefixes]
    sim_prefixes = [p.lower() for p in sim_prefixes]

    def classify(d: str) -> str:
        dl = str(d).lower()
        if any(dl.startswith(p) for p in real_prefixes):
            return "real"
        if any(dl.startswith(p) for p in sim_prefixes):
            return "synthetic"
        return "unknown"

    return np.array([classify(d) for d in dataset_labels], dtype=object)


def color_by_sim_real(dataset_labels: np.ndarray, real_prefixes: Sequence[str],
                       sim_prefixes: Sequence[str]) -> Tuple[np.ndarray, Dict[str, tuple]]:
    """Colors by real vs. synthetic source, determined via configurable
    prefix-matching on the `dataset` field (e.g. 'real-fda-cdcl3' ->
    'real', 'syn-fda' -> 'synthetic'). Anything matching neither prefix
    list falls into 'unknown' -- check that bucket's size if it's large,
    since it likely means the prefixes need adjusting for your naming
    convention."""
    categories = classify_sim_real(dataset_labels, real_prefixes, sim_prefixes)
    fixed_palette = {
        "real": (0.20, 0.45, 0.85, 1.0),
        "synthetic": (0.90, 0.45, 0.10, 1.0),
        "unknown": tuple(GRAY),
    }
    present = [c for c in ("real", "synthetic", "unknown") if c in set(categories.tolist())]
    palette = {c: fixed_palette[c] for c in present}
    colors = np.array([palette[c] for c in categories])
    return colors, palette


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def _scatter_colors(color_by: str, dataset_labels, cluster_labels, np_class_labels,
                     highlight_clusters, np_class_top_n, real_prefixes, sim_prefixes):
    """Shared color-resolution logic used by both the main scatter and the
    per-region zoom panels, so all panels use identical color assignments."""
    if color_by == "cluster":
        labels_plot, cmap, norm, _ = color_by_cluster(cluster_labels, highlight_clusters)
        return {"mode": "cluster", "c": labels_plot, "cmap": cmap, "norm": norm}
    elif color_by == "dataset":
        colors, palette = color_by_dataset(dataset_labels)
        return {"mode": "flat", "c": colors, "palette": palette}
    elif color_by == "np_class":
        colors, palette = color_by_np_class(np_class_labels, np_class_top_n)
        return {"mode": "flat", "c": colors, "palette": palette}
    elif color_by == "sim_real":
        colors, palette = color_by_sim_real(dataset_labels, real_prefixes, sim_prefixes)
        return {"mode": "flat", "c": colors, "palette": palette}
    else:
        return {"mode": "none"}


def _draw_scatter(ax, emb2d, resolved, s, alpha, with_legend):
    if resolved["mode"] == "cluster":
        ax.scatter(emb2d[:, 0], emb2d[:, 1], c=resolved["c"], cmap=resolved["cmap"],
                   norm=resolved["norm"], s=s, alpha=alpha, linewidths=0)
    elif resolved["mode"] == "flat":
        ax.scatter(emb2d[:, 0], emb2d[:, 1], c=resolved["c"], s=s, alpha=alpha, linewidths=0)
        if with_legend:
            handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=col,
                                   markersize=8, label=name)
                       for name, col in resolved["palette"].items()]
            ax.legend(handles=handles, loc="best", frameon=True, fontsize=9)
    else:
        ax.scatter(emb2d[:, 0], emb2d[:, 1], c="steelblue", s=s, alpha=alpha, linewidths=0)


def plot_main_scatter(emb2d: np.ndarray, color_by: str, dataset_labels: np.ndarray,
                       cluster_labels: Optional[np.ndarray], np_class_labels: Optional[np.ndarray],
                       highlight_clusters: Optional[Sequence[int]], np_class_top_n: int,
                       real_prefixes: Sequence[str], sim_prefixes: Sequence[str],
                       regions: List[Region], out_path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(10, 10))

    resolved = _scatter_colors(color_by, dataset_labels, cluster_labels, np_class_labels,
                                highlight_clusters, np_class_top_n, real_prefixes, sim_prefixes)
    _draw_scatter(ax, emb2d, resolved, s=0.25, alpha=0.25, with_legend=True)

    # Highlight overlays (rectangles + labels for clusters and/or SMILES regions)
    for region in regions:
        half = WINDOW_SIZE_GLOBAL / 2
        rect = Rectangle(
            (region.center[0] - half, region.center[1] - half),
            WINDOW_SIZE_GLOBAL, WINDOW_SIZE_GLOBAL,
            fill=False, linewidth=2, edgecolor=region.color,
        )
        ax.add_patch(rect)

        marker = "*" if region.kind == "smiles" else None
        if marker:
            ax.scatter([region.center[0]], [region.center[1]], marker=marker, s=120,
                       facecolor=region.color, edgecolor="black", linewidths=0.5, zorder=6)

        ax.text(region.center[0], region.center[1] + half + 0.2, region.label,
                color=region.color, ha="center", fontsize=10, weight="bold")

    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_region_panel(region: Region, emb2d: np.ndarray, color_by: str,
                       dataset_labels: np.ndarray, cluster_labels: Optional[np.ndarray],
                       np_class_labels: Optional[np.ndarray], highlight_clusters: Optional[Sequence[int]],
                       np_class_top_n: int, real_prefixes: Sequence[str], sim_prefixes: Sequence[str],
                       smiles: List[str], n_mols: int, mols_per_row: int, seed: int,
                       out_dir: Path, dpi: int) -> None:
    fig = plt.figure(figsize=(8, 4))
    gs = GridSpec(1, 2, width_ratios=[1, 1], figure=fig)

    ax_scatter = fig.add_subplot(gs[0, 0])
    resolved = _scatter_colors(color_by, dataset_labels, cluster_labels, np_class_labels,
                                highlight_clusters, np_class_top_n, real_prefixes, sim_prefixes)
    _draw_scatter(ax_scatter, emb2d, resolved, s=1, alpha=0.8, with_legend=False)

    half = WINDOW_SIZE_GLOBAL / 2
    ax_scatter.set_xlim(region.center[0] - half, region.center[0] + half)
    ax_scatter.set_ylim(region.center[1] - half, region.center[1] + half)
    ax_scatter.set_aspect("equal")
    ax_scatter.set_xticks([])
    ax_scatter.set_yticks([])
    title = f"Cluster {region.region_id}" if region.kind == "cluster" else f"{region.label}: query molecule"
    ax_scatter.set_title(title)

    mols, legends = sample_region_smiles(region, smiles, n_mols, seed)
    mol_img = mols_to_image(mols, legends, mols_per_row=mols_per_row)

    ax_mols = fig.add_subplot(gs[0, 1])
    if mol_img is not None:
        ax_mols.imshow(mol_img)
    else:
        ax_mols.text(0.5, 0.5, "no valid SMILES\nsampled", ha="center", va="center",
                      transform=ax_mols.transAxes)
    ax_mols.set_xticks([])
    ax_mols.set_yticks([])
    ax_mols.set_title("sampled molecules")

    plt.tight_layout()

    region_tag = region.region_id if region.kind == "cluster" else region.label
    out_path = out_dir / f"region_{region.kind}_{region_tag}.png"
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    if mol_img is not None:
        mol_img.save(out_dir / f"region_{region.kind}_{region_tag}_mols.png")

    print(f"Saved {out_path}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generalized UMAP main-figure plot with optional dataset "
                    "subsetting, cluster highlighting, SMILES-of-interest "
                    "highlighting, and NP-class / sim-vs-real coloring.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--prefix", type=str, default="cotrain")
    p.add_argument("--splits", nargs="+", default=["train"], choices=["train", "val", "test"])
    p.add_argument("--embedding-key", type=str, default="global_cond")

    p.add_argument("--datasets", nargs="+", default=None,
                   help="Restrict to these source datasets (e.g. spectranp "
                        "nmrexp). Default: use everything loaded.")

    p.add_argument("--cluster-labels-path", type=Path, default=None,
                   help="cotrain_cluster_labs.npy from cluster_cotrain_molecules.py "
                        "(required for --color-by cluster or --highlight-clusters).")
    p.add_argument("--np-class-labels-path", type=Path, default=None,
                   help="labels.pkl from npclassifier_local.py: List[str] of "
                        "NPClassifier's dominant class per molecule, aligned to "
                        "the unfiltered data (required for --color-by np_class).")
    p.add_argument("--np-class-probs-path", type=Path, default=None,
                   help="labels_probs.pkl from npclassifier_local.py: List[float] "
                        "matching --np-class-labels-path 1:1. If given, molecules "
                        "below --np-class-min-probability are colored gray ('Other') "
                        "regardless of their assigned class.")
    p.add_argument("--np-class-min-probability", type=float, default=0.0,
                   help="Only takes effect if --np-class-probs-path is given. "
                        "Molecules with dominant_probability below this are "
                        "grayed out instead of colored by class.")
    p.add_argument("--np-class-top-n", type=int, default=10,
                   help="Number of most common NP classes to color distinctly; "
                        "the rest are bucketed into a gray 'Other'.")
    p.add_argument("--real-prefixes", nargs="+", default=["real"],
                   help="Case-insensitive prefixes of `dataset` values counted as 'real' "
                        "(for --color-by sim_real).")
    p.add_argument("--sim-prefixes", nargs="+", default=["syn"],
                   help="Case-insensitive prefixes of `dataset` values counted as "
                        "'synthetic' (for --color-by sim_real).")

    p.add_argument("--color-by", choices=["dataset", "cluster", "np_class", "sim_real", "none"],
                   default="dataset")
    p.add_argument("--highlight-clusters", nargs="+", type=int, default=None,
                   help="Cluster IDs to rectangle-highlight + inset. Only "
                        "drawn if explicitly given.")

    p.add_argument("--smiles-of-interest", nargs="+", default=None,
                   help="SMILES strings to star-highlight + inset (matched "
                        "against the loaded data via canonical SMILES).")

    p.add_argument("--umap-embedding-path", type=Path, default=None,
                   help="Precomputed 2D projection (.npy), aligned to the "
                        "exact subset being plotted. If omitted, UMAP is fit "
                        "fresh on --embedding-key.")
    p.add_argument("--save-umap-embedding", type=Path, default=None,
                   help="Where to cache a freshly-fit 2D projection.")
    p.add_argument("--umap-n-neighbors", type=int, default=30)
    p.add_argument("--umap-min-dist", type=float, default=0.1)
    p.add_argument("--umap-metric", type=str, default="euclidean")
    p.add_argument("--pca-dim", type=int, default=50,
                   help="Reduce to this many dimensions via PCA before fitting UMAP "
                        "(only applies when fitting fresh, not when loading a "
                        "precomputed --umap-embedding-path). Set to 0 to disable.")

    p.add_argument("--window-size", type=float, default=2.5)
    p.add_argument("--n-mols-per-region", type=int, default=6)
    p.add_argument("--mols-per-row", type=int, default=2)

    p.add_argument("--out-dir", type=Path, default=Path("umap_figure"))
    p.add_argument("--main-fig-name", type=str, default="umap_main.png")
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    global WINDOW_SIZE_GLOBAL

    args = parse_args(argv)
    WINDOW_SIZE_GLOBAL = args.window_size
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.color_by == "cluster" and args.cluster_labels_path is None:
        raise ValueError("--color-by cluster requires --cluster-labels-path.")
    if args.highlight_clusters and args.cluster_labels_path is None:
        raise ValueError("--highlight-clusters requires --cluster-labels-path.")
    if args.color_by == "np_class" and args.np_class_labels_path is None:
        raise ValueError("--color-by np_class requires --np-class-labels-path.")

    print("=" * 78)
    print("UMAP main figure")
    print(f"  data dir           : {args.data_dir}")
    print(f"  embedding key      : {args.embedding_key}")
    print(f"  datasets           : {args.datasets or 'ALL'}")
    print(f"  color-by           : {args.color_by}")
    print(f"  highlight clusters : {args.highlight_clusters}")
    print(f"  smiles of interest : {args.smiles_of_interest}")
    print("=" * 78)

    data = load_cotrain_data(args.data_dir, args.prefix, args.splits, args.embedding_key)
    n_total = len(data["smiles"])

    cluster_labels_full = None
    if args.cluster_labels_path is not None:
        cluster_labels_full = np.load(args.cluster_labels_path)
        if len(cluster_labels_full) != n_total:
            raise ValueError(
                f"{args.cluster_labels_path} has {len(cluster_labels_full)} labels "
                f"but the loaded (unfiltered) data has {n_total} molecules -- check "
                f"--splits matches what was used to generate the cluster labels."
            )

    np_class_labels_full = None
    if args.np_class_labels_path is not None:
        np_class_labels_full = load_pickle_labels(args.np_class_labels_path, n_total, "NP-class")

    np_class_probs_full = None
    if args.np_class_probs_path is not None:
        np_class_probs_full = load_pickle_labels(
            args.np_class_probs_path, n_total, "NP-class probability"
        ).astype(float)

    data, filtered_extras = filter_by_datasets(
        data, args.datasets,
        {"cluster": cluster_labels_full, "np_class": np_class_labels_full,
         "np_class_probs": np_class_probs_full},
    )
    cluster_labels = filtered_extras["cluster"]
    np_class_labels = filtered_extras["np_class"]
    np_class_probs = filtered_extras["np_class_probs"]

    if np_class_labels is not None and np_class_probs is not None:
        low_conf = np_class_probs < args.np_class_min_probability
        n_low_conf = int(low_conf.sum())
        if n_low_conf > 0:
            print(f"Graying out {n_low_conf}/{len(np_class_labels)} molecules with "
                  f"dominant_probability < {args.np_class_min_probability}")
            np_class_labels = np_class_labels.copy()
            np_class_labels[low_conf] = "Other"

    smiles = data["smiles"]
    embedding = data["embedding"]
    dataset_labels = data["dataset"]

    emb2d = load_or_compute_umap(
        embedding, args.umap_embedding_path, args.save_umap_embedding,
        args.umap_n_neighbors, args.umap_min_dist, args.umap_metric, args.seed,
        pca_dim=args.pca_dim,
    )

    regions: List[Region] = []
    if args.highlight_clusters:
        regions.extend(build_cluster_regions(args.highlight_clusters, cluster_labels, emb2d))
    if args.smiles_of_interest:
        matches = match_smiles_of_interest(args.smiles_of_interest, smiles)
        regions.extend(build_smiles_regions(matches, emb2d, args.window_size))

    plot_main_scatter(
        emb2d, args.color_by, dataset_labels, cluster_labels, np_class_labels,
        args.highlight_clusters, args.np_class_top_n, args.real_prefixes, args.sim_prefixes,
        regions, args.out_dir / args.main_fig_name, args.dpi,
    )

    if regions:
        for region in regions:
            plot_region_panel(
                region, emb2d, args.color_by, dataset_labels, cluster_labels, np_class_labels,
                args.highlight_clusters, args.np_class_top_n, args.real_prefixes, args.sim_prefixes,
                smiles, args.n_mols_per_region, args.mols_per_row, args.seed, args.out_dir, args.dpi,
            )

    manifest = {
        "data_dir": str(args.data_dir), "prefix": args.prefix, "splits": args.splits,
        "embedding_key": args.embedding_key,
        "datasets": args.datasets, "color_by": args.color_by,
        "highlight_clusters": args.highlight_clusters,
        "smiles_of_interest": args.smiles_of_interest,
        "n_molecules_plotted": len(smiles), "window_size": args.window_size,
        "regions": [
            {"kind": r.kind, "id": str(r.region_id), "label": r.label, "center": r.center}
            for r in regions
        ],
    }
    (args.out_dir / "umap_figure_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Done.")


if __name__ == "__main__":
    main()


# #!/usr/bin/env python
# """
# umap_main_figure.py
# ======================

# Generalized rewrite of the original "2a_pretty_plot" UMAP figure script.
# Preserves its exact visual style (rainbow cluster colors + gray "other",
# thin zoom rectangles with matching-color cluster-id labels, side-by-side
# scatter-zoom + molecule-grid inset panels) while adding:

#     1. Dataset subsetting -- plot only e.g. {spectranp} or {spectranp,
#        nmrexp} instead of always all three cotrain sources.
#     2. SMILES-of-interest highlighting -- give a list of SMILES strings and
#        the script finds each in the loaded data (via canonical-SMILES
#        match), marks it with a star on the main scatter, and saves a
#        zoom + molecule-grid inset centered on it (neighborhood molecules,
#        with the query molecule labeled "query" in its grid legend).
#     3. Cluster highlighting is now fully optional -- clusters are only
#        colored/gray-masked/inset if you pass `--highlight-clusters`,
#        instead of the original's hardcoded `clusters_to_show`.
#     4. NEW: --color-by np_class -- color by NPClassifier's dominant
#        natural-product class per molecule (top-N most common classes,
#        everything else bucketed into a gray "Other").
#     5. NEW: --color-by sim_real -- color by whether each molecule's
#        `dataset` source is real vs. synthetic, via configurable prefix
#        matching (--real-prefixes / --sim-prefixes), with an "unknown"
#        bucket for anything that matches neither.

# Background coloring of the full scatter (`--color-by`) is independent of
# the highlight overlays (clusters / SMILES-of-interest), so you can e.g.
# color by NP class while still drawing cluster-zoom insets on top.

# Inputs
# ------
#     - `<prefix>_<split>_global_cond.pt`  (extract_cotrain_embeddings.py)
#     - `cotrain_cluster_labs.npy`          (cluster_cotrain_molecules.py;
#                                             only needed for --color-by
#                                             cluster or --highlight-clusters)
#     - `labels.pkl`                        (npclassifier_local.py; only
#                                             needed for --color-by np_class)
#     - a precomputed 2D projection (--umap-embedding-path), OR the script
#       fits UMAP itself on the (possibly subsetted) embeddings and can cache
#       the result via --save-umap-embedding.

# Usage
# -----
#     # Reproduce the original figure: all 3 sources, 5 highlighted clusters
#     python umap_main_figure.py \\
#         --data-dir /scratch/gpfs/ZHONGE/jc4587/nmr_embs_cotrain \\
#         --cluster-labels-path cluster_summary/cotrain_cluster_labs.npy \\
#         --color-by cluster --highlight-clusters 0 48 99 149 199 \\
#         --out-dir umap_figure

#     # Color by NPClassifier's top-10 most common classes
#     python umap_main_figure.py \\
#         --data-dir /scratch/gpfs/ZHONGE/jc4587/nmr_embs_cotrain \\
#         --np-class-labels-path labels.pkl --color-by np_class \\
#         --np-class-top-n 10 --out-dir umap_figure_npclass

#     # Color by real vs. synthetic source
#     python umap_main_figure.py \\
#         --data-dir /scratch/gpfs/ZHONGE/jc4587/nmr_embs_cotrain \\
#         --color-by sim_real --out-dir umap_figure_simreal
# """

# from __future__ import annotations

# import argparse
# import json
# import pickle
# import random
# from pathlib import Path
# from typing import Dict, List, Optional, Sequence, Tuple

# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap, BoundaryNorm
# from matplotlib.patches import Rectangle
# from matplotlib.gridspec import GridSpec

# from rdkit import Chem
# from rdkit.Chem import Draw
# from rdkit import RDLogger

# RDLogger.DisableLog("rdApp.*")

# QUALITATIVE_PALETTE = plt.cm.tab10(np.linspace(0, 1, 10))
# GRAY = np.array([0.75, 0.75, 0.75, 1.0])

# WINDOW_SIZE_GLOBAL = 2.5  # set from --window-size in main(); module-level for plot fns


# # -----------------------------------------------------------------------------
# # Data loading
# # -----------------------------------------------------------------------------

# def load_cotrain_data(data_dir: Path, prefix: str, splits: Sequence[str],
#                        embedding_key: str) -> Dict:
#     smiles: List[str] = []
#     dataset: List[str] = []
#     embed_parts: List[torch.Tensor] = []

#     for split in splits:
#         path = data_dir / f"{prefix}_{split}_global_cond.pt"
#         if not path.exists():
#             raise FileNotFoundError(f"Expected extraction output not found: {path}")
#         print(f"Loading {path} ...")
#         d = torch.load(path, map_location="cpu")
#         if embedding_key not in d:
#             raise KeyError(f"'{embedding_key}' not found in {path}. Keys: {list(d.keys())}")
#         smiles.extend(d["smiles"])
#         dataset.extend(d["dataset"])
#         embed_parts.append(d[embedding_key])

#     embedding = torch.cat(embed_parts, dim=0).numpy().astype(np.float32)
#     return {"smiles": smiles, "dataset": np.asarray(dataset), "embedding": embedding}


# def load_pickle_labels(path: Path, expected_len: int, label_name: str) -> np.ndarray:
#     """Loads a pickled List[str]/List[float] (e.g. labels.pkl from
#     npclassifier_local.py) and checks it's aligned to the unfiltered data."""
#     with open(path, "rb") as f:
#         labels = pickle.load(f)
#     labels = np.asarray(labels, dtype=object)
#     if len(labels) != expected_len:
#         raise ValueError(
#             f"{path} has {len(labels)} {label_name} entries but the loaded "
#             f"(unfiltered) data has {expected_len} molecules -- check that "
#             f"--splits matches whatever split(s) {label_name} was generated for "
#             f"(e.g. npclassifier_local.py labels.pkl is typically train-only)."
#         )
#     return labels


# def filter_by_datasets(data: Dict, datasets: Optional[Sequence[str]],
#                         extra_label_arrays: Dict[str, Optional[np.ndarray]]
#                         ) -> Tuple[Dict, Dict[str, Optional[np.ndarray]]]:
#     if datasets is None:
#         return data, extra_label_arrays

#     mask = np.isin(data["dataset"], list(datasets))
#     n_before, n_after = len(data["dataset"]), int(mask.sum())
#     if n_after == 0:
#         raise ValueError(f"No molecules found for --datasets {datasets}. "
#                           f"Available sources: {sorted(set(data['dataset'].tolist()))}")
#     print(f"Subsetting to datasets={list(datasets)}: {n_before} -> {n_after} molecules")

#     filtered = {
#         "smiles": [s for s, m in zip(data["smiles"], mask) if m],
#         "dataset": data["dataset"][mask],
#         "embedding": data["embedding"][mask],
#     }
#     filtered_extras = {
#         name: (arr[mask] if arr is not None else None)
#         for name, arr in extra_label_arrays.items()
#     }
#     return filtered, filtered_extras


# # -----------------------------------------------------------------------------
# # UMAP (load-or-compute, cacheable)
# # -----------------------------------------------------------------------------

# def load_or_compute_umap(embedding: np.ndarray, load_path: Optional[Path],
#                           save_path: Optional[Path], n_neighbors: int,
#                           min_dist: float, metric: str, seed: int) -> np.ndarray:
#     if load_path is not None:
#         print(f"Loading precomputed 2D projection from {load_path}")
#         emb2d = np.load(load_path)
#         if emb2d.shape[0] != embedding.shape[0]:
#             raise ValueError(
#                 f"{load_path} has {emb2d.shape[0]} rows but the (possibly "
#                 f"subsetted) loaded data has {embedding.shape[0]} molecules "
#                 f"-- a precomputed projection must be aligned to the exact "
#                 f"subset being plotted."
#             )
#         return emb2d

#     import umap.umap_ as umap_lib

#     print(f"Fitting UMAP (n_neighbors={n_neighbors}, min_dist={min_dist}, "
#           f"metric={metric}) on {embedding.shape[0]} points ...")
#     reducer = umap_lib.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
#                              metric=metric, random_state=seed)
#     emb2d = reducer.fit_transform(embedding)

#     if save_path is not None:
#         save_path.parent.mkdir(parents=True, exist_ok=True)
#         np.save(save_path, emb2d)
#         print(f"Cached 2D projection to {save_path}")

#     return emb2d


# # -----------------------------------------------------------------------------
# # SMILES-of-interest matching
# # -----------------------------------------------------------------------------

# def canonicalize(smi: str) -> Optional[str]:
#     mol = Chem.MolFromSmiles(smi)
#     if mol is None:
#         return None
#     return Chem.MolToSmiles(mol)


# def match_smiles_of_interest(query_smiles: Sequence[str],
#                               dataset_smiles: List[str]) -> Dict[str, int]:
#     """Returns {query_smiles: index_in_dataset}, skipping (with a warning)
#     any query that doesn't canonicalize or isn't found in the dataset."""
#     canon_to_idx: Dict[str, int] = {}
#     for i, s in enumerate(dataset_smiles):
#         c = canonicalize(s)
#         if c is not None and c not in canon_to_idx:
#             canon_to_idx[c] = i

#     matches = {}
#     for q in query_smiles:
#         cq = canonicalize(q)
#         if cq is None:
#             print(f"[warn] Could not parse query SMILES, skipping: {q!r}")
#             continue
#         if cq not in canon_to_idx:
#             print(f"[warn] Query SMILES not found in the loaded/subsetted "
#                   f"data, skipping: {q!r}")
#             continue
#         matches[q] = canon_to_idx[cq]
#     return matches


# # -----------------------------------------------------------------------------
# # Region definitions (clusters and/or SMILES-of-interest)
# # -----------------------------------------------------------------------------

# class Region:
#     def __init__(self, kind: str, region_id, label: str, color, center: Tuple[float, float],
#                  member_idx: np.ndarray, query_idx: Optional[int] = None):
#         self.kind = kind                # "cluster" | "smiles"
#         self.region_id = region_id      # cluster int id, or the raw query smiles string
#         self.label = label              # text drawn on the main plot
#         self.color = color
#         self.center = center
#         self.member_idx = member_idx    # indices used to sample the molecule-grid panel
#         self.query_idx = query_idx      # for "smiles" regions: the exact matched index


# def build_cluster_regions(cluster_ids: Sequence[int], labels: np.ndarray,
#                            emb2d: np.ndarray) -> List[Region]:
#     n_normal = int(labels.max()) + 1
#     base_colors = plt.cm.rainbow(np.linspace(0, 1, n_normal))

#     regions = []
#     for cid in cluster_ids:
#         cluster_mask = labels == cid
#         pts = emb2d[cluster_mask]
#         if len(pts) == 0:
#             print(f"[warn] Cluster {cid} has no members in the current "
#                   f"(possibly subsetted) data, skipping.")
#             continue
#         center = (float(np.median(pts[:, 0])), float(np.median(pts[:, 1])))
#         member_idx = np.where(cluster_mask)[0]
#         color = base_colors[cid % n_normal]
#         regions.append(Region("cluster", cid, str(cid), color, center, member_idx))
#     return regions


# def build_smiles_regions(matches: Dict[str, int], emb2d: np.ndarray,
#                           window_size: float) -> List[Region]:
#     regions = []
#     for i, (query, idx) in enumerate(matches.items()):
#         center = (float(emb2d[idx, 0]), float(emb2d[idx, 1]))
#         half = window_size / 2
#         in_window = (
#             (emb2d[:, 0] >= center[0] - half) & (emb2d[:, 0] <= center[0] + half) &
#             (emb2d[:, 1] >= center[1] - half) & (emb2d[:, 1] <= center[1] + half)
#         )
#         member_idx = np.where(in_window)[0]
#         color = QUALITATIVE_PALETTE[i % len(QUALITATIVE_PALETTE)]
#         label = f"S{i + 1}"
#         regions.append(Region("smiles", query, label, color, center, member_idx, query_idx=idx))
#     return regions


# # -----------------------------------------------------------------------------
# # Molecule sampling + rendering (same primitives as the original script)
# # -----------------------------------------------------------------------------

# def sample_region_smiles(region: Region, smiles: List[str], n_mols: int,
#                           seed: int) -> Tuple[List, List[str]]:
#     """Returns (rdkit_mols, legends). For 'smiles' regions the query molecule
#     is always included first and legended 'query'."""
#     rng = random.Random(seed + (region.query_idx if region.query_idx is not None
#                                  else hash(region.region_id) % 10_000))

#     ordered_idx = list(region.member_idx)
#     rng.shuffle(ordered_idx)

#     chosen_idx = []
#     if region.kind == "smiles" and region.query_idx is not None:
#         chosen_idx.append(region.query_idx)

#     for idx in ordered_idx:
#         if len(chosen_idx) >= n_mols:
#             break
#         if idx in chosen_idx:
#             continue
#         if Chem.MolFromSmiles(smiles[idx]) is None:
#             continue
#         chosen_idx.append(idx)

#     mols, legends = [], []
#     for idx in chosen_idx:
#         mol = Chem.MolFromSmiles(smiles[idx])
#         if mol is None:
#             continue
#         mols.append(mol)
#         legends.append("query" if (region.kind == "smiles" and idx == region.query_idx) else "")
#     return mols, legends


# def mols_to_image(mols: List, legends: Optional[List[str]] = None,
#                    mols_per_row: int = 2, sub_img_size: Tuple[int, int] = (180, 180)):
#     if len(mols) == 0:
#         return None
#     return Draw.MolsToGridImage(
#         mols, molsPerRow=mols_per_row, subImgSize=sub_img_size,
#         legends=legends, returnPNG=False,
#     )


# # -----------------------------------------------------------------------------
# # Background coloring for the full scatter
# # -----------------------------------------------------------------------------

# def color_by_dataset(dataset_labels: np.ndarray) -> Tuple[np.ndarray, Dict[str, tuple]]:
#     sources = sorted(set(dataset_labels.tolist()))
#     palette = {src: QUALITATIVE_PALETTE[i % len(QUALITATIVE_PALETTE)] for i, src in enumerate(sources)}
#     colors = np.array([palette[s] for s in dataset_labels])
#     return colors, palette


# def color_by_cluster(labels: np.ndarray, highlight_clusters: Optional[Sequence[int]]
#                       ) -> Tuple[np.ndarray, ListedColormap, BoundaryNorm, np.ndarray]:
#     n_normal = int(labels.max()) + 1
#     base_colors = plt.cm.rainbow(np.linspace(0, 1, n_normal))
#     gray = np.array([[0.75, 0.75, 0.75, 1.0]])
#     custom_cmap = ListedColormap(np.vstack([base_colors, gray]))

#     labels_plot = labels.copy()
#     special_label = n_normal  # == labels.max() + 1

#     if highlight_clusters is not None:
#         mask = ~np.isin(labels_plot, list(highlight_clusters))
#         labels_plot[mask] = special_label

#     bounds = np.arange(special_label + 2) - 0.5
#     norm = BoundaryNorm(bounds, custom_cmap.N)
#     return labels_plot, custom_cmap, norm, base_colors


# def _bucket_by_frequency(labels: np.ndarray, top_n: int, other_label: str = "Other"
#                           ) -> np.ndarray:
#     """Keeps the top_n most frequent distinct values as-is; everything else
#     becomes `other_label`."""
#     vals, counts = np.unique(labels, return_counts=True)
#     order = np.argsort(-counts)
#     keep = set(vals[order][:top_n].tolist())
#     return np.array([v if v in keep else other_label for v in labels], dtype=object)


# def color_by_np_class(np_class_labels: np.ndarray, top_n: int
#                        ) -> Tuple[np.ndarray, Dict[str, tuple]]:
#     """Colors by NPClassifier's dominant class per molecule, keeping only the
#     top_n most common classes distinctly colored and bucketing everything
#     else (long-tail classes) into a gray 'Other'."""
#     bucketed = _bucket_by_frequency(np_class_labels, top_n, other_label="Other")
#     categories = sorted(c for c in set(bucketed.tolist()) if c != "Other")

#     palette = {cat: QUALITATIVE_PALETTE[i % len(QUALITATIVE_PALETTE)] for i, cat in enumerate(categories)}
#     if "Other" in bucketed:
#         palette["Other"] = tuple(GRAY)

#     colors = np.array([palette[c] for c in bucketed])
#     return colors, palette


# def classify_sim_real(dataset_labels: np.ndarray, real_prefixes: Sequence[str],
#                        sim_prefixes: Sequence[str]) -> np.ndarray:
#     real_prefixes = [p.lower() for p in real_prefixes]
#     sim_prefixes = [p.lower() for p in sim_prefixes]

#     def classify(d: str) -> str:
#         dl = str(d).lower()
#         if any(dl.startswith(p) for p in real_prefixes):
#             return "real"
#         if any(dl.startswith(p) for p in sim_prefixes):
#             return "synthetic"
#         return "unknown"

#     return np.array([classify(d) for d in dataset_labels], dtype=object)


# def color_by_sim_real(dataset_labels: np.ndarray, real_prefixes: Sequence[str],
#                        sim_prefixes: Sequence[str]) -> Tuple[np.ndarray, Dict[str, tuple]]:
#     """Colors by real vs. synthetic source, determined via configurable
#     prefix-matching on the `dataset` field (e.g. 'real-fda-cdcl3' ->
#     'real', 'syn-fda' -> 'synthetic'). Anything matching neither prefix
#     list falls into 'unknown' -- check that bucket's size if it's large,
#     since it likely means the prefixes need adjusting for your naming
#     convention."""
#     categories = classify_sim_real(dataset_labels, real_prefixes, sim_prefixes)
#     fixed_palette = {
#         "real": (0.20, 0.45, 0.85, 1.0),
#         "synthetic": (0.90, 0.45, 0.10, 1.0),
#         "unknown": tuple(GRAY),
#     }
#     present = [c for c in ("real", "synthetic", "unknown") if c in set(categories.tolist())]
#     palette = {c: fixed_palette[c] for c in present}
#     colors = np.array([palette[c] for c in categories])
#     return colors, palette


# # -----------------------------------------------------------------------------
# # Plotting
# # -----------------------------------------------------------------------------

# def _scatter_colors(color_by: str, dataset_labels, cluster_labels, np_class_labels,
#                      highlight_clusters, np_class_top_n, real_prefixes, sim_prefixes):
#     """Shared color-resolution logic used by both the main scatter and the
#     per-region zoom panels, so all panels use identical color assignments."""
#     if color_by == "cluster":
#         labels_plot, cmap, norm, _ = color_by_cluster(cluster_labels, highlight_clusters)
#         return {"mode": "cluster", "c": labels_plot, "cmap": cmap, "norm": norm}
#     elif color_by == "dataset":
#         colors, palette = color_by_dataset(dataset_labels)
#         return {"mode": "flat", "c": colors, "palette": palette}
#     elif color_by == "np_class":
#         colors, palette = color_by_np_class(np_class_labels, np_class_top_n)
#         return {"mode": "flat", "c": colors, "palette": palette}
#     elif color_by == "sim_real":
#         colors, palette = color_by_sim_real(dataset_labels, real_prefixes, sim_prefixes)
#         return {"mode": "flat", "c": colors, "palette": palette}
#     else:
#         return {"mode": "none"}


# def _draw_scatter(ax, emb2d, resolved, s, alpha, with_legend):
#     if resolved["mode"] == "cluster":
#         ax.scatter(emb2d[:, 0], emb2d[:, 1], c=resolved["c"], cmap=resolved["cmap"],
#                    norm=resolved["norm"], s=s, alpha=alpha, linewidths=0)
#     elif resolved["mode"] == "flat":
#         ax.scatter(emb2d[:, 0], emb2d[:, 1], c=resolved["c"], s=s, alpha=alpha, linewidths=0)
#         if with_legend:
#             handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=col,
#                                    markersize=8, label=name)
#                        for name, col in resolved["palette"].items()]
#             ax.legend(handles=handles, loc="best", frameon=True, fontsize=9)
#     else:
#         ax.scatter(emb2d[:, 0], emb2d[:, 1], c="steelblue", s=s, alpha=alpha, linewidths=0)


# def plot_main_scatter(emb2d: np.ndarray, color_by: str, dataset_labels: np.ndarray,
#                        cluster_labels: Optional[np.ndarray], np_class_labels: Optional[np.ndarray],
#                        highlight_clusters: Optional[Sequence[int]], np_class_top_n: int,
#                        real_prefixes: Sequence[str], sim_prefixes: Sequence[str],
#                        regions: List[Region], out_path: Path, dpi: int) -> None:
#     fig, ax = plt.subplots(figsize=(10, 10))

#     resolved = _scatter_colors(color_by, dataset_labels, cluster_labels, np_class_labels,
#                                 highlight_clusters, np_class_top_n, real_prefixes, sim_prefixes)
#     _draw_scatter(ax, emb2d, resolved, s=0.25, alpha=0.25, with_legend=True)

#     # Highlight overlays (rectangles + labels for clusters and/or SMILES regions)
#     for region in regions:
#         half = WINDOW_SIZE_GLOBAL / 2
#         rect = Rectangle(
#             (region.center[0] - half, region.center[1] - half),
#             WINDOW_SIZE_GLOBAL, WINDOW_SIZE_GLOBAL,
#             fill=False, linewidth=2, edgecolor=region.color,
#         )
#         ax.add_patch(rect)

#         marker = "*" if region.kind == "smiles" else None
#         if marker:
#             ax.scatter([region.center[0]], [region.center[1]], marker=marker, s=120,
#                        facecolor=region.color, edgecolor="black", linewidths=0.5, zorder=6)

#         ax.text(region.center[0], region.center[1] + half + 0.2, region.label,
#                 color=region.color, ha="center", fontsize=10, weight="bold")

#     ax.set_xlabel("UMAP-1")
#     ax.set_ylabel("UMAP-2")
#     plt.tight_layout()
#     plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
#     plt.close(fig)
#     print(f"Saved {out_path}")


# def plot_region_panel(region: Region, emb2d: np.ndarray, color_by: str,
#                        dataset_labels: np.ndarray, cluster_labels: Optional[np.ndarray],
#                        np_class_labels: Optional[np.ndarray], highlight_clusters: Optional[Sequence[int]],
#                        np_class_top_n: int, real_prefixes: Sequence[str], sim_prefixes: Sequence[str],
#                        smiles: List[str], n_mols: int, mols_per_row: int, seed: int,
#                        out_dir: Path, dpi: int) -> None:
#     fig = plt.figure(figsize=(8, 4))
#     gs = GridSpec(1, 2, width_ratios=[1, 1], figure=fig)

#     ax_scatter = fig.add_subplot(gs[0, 0])
#     resolved = _scatter_colors(color_by, dataset_labels, cluster_labels, np_class_labels,
#                                 highlight_clusters, np_class_top_n, real_prefixes, sim_prefixes)
#     _draw_scatter(ax_scatter, emb2d, resolved, s=1, alpha=0.8, with_legend=False)

#     half = WINDOW_SIZE_GLOBAL / 2
#     ax_scatter.set_xlim(region.center[0] - half, region.center[0] + half)
#     ax_scatter.set_ylim(region.center[1] - half, region.center[1] + half)
#     ax_scatter.set_aspect("equal")
#     ax_scatter.set_xticks([])
#     ax_scatter.set_yticks([])
#     title = f"Cluster {region.region_id}" if region.kind == "cluster" else f"{region.label}: query molecule"
#     ax_scatter.set_title(title)

#     mols, legends = sample_region_smiles(region, smiles, n_mols, seed)
#     mol_img = mols_to_image(mols, legends, mols_per_row=mols_per_row)

#     ax_mols = fig.add_subplot(gs[0, 1])
#     if mol_img is not None:
#         ax_mols.imshow(mol_img)
#     else:
#         ax_mols.text(0.5, 0.5, "no valid SMILES\nsampled", ha="center", va="center",
#                       transform=ax_mols.transAxes)
#     ax_mols.set_xticks([])
#     ax_mols.set_yticks([])
#     ax_mols.set_title("sampled molecules")

#     plt.tight_layout()

#     region_tag = region.region_id if region.kind == "cluster" else region.label
#     out_path = out_dir / f"region_{region.kind}_{region_tag}.png"
#     plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
#     plt.close(fig)

#     if mol_img is not None:
#         mol_img.save(out_dir / f"region_{region.kind}_{region_tag}_mols.png")

#     print(f"Saved {out_path}")


# # -----------------------------------------------------------------------------
# # CLI
# # -----------------------------------------------------------------------------

# def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
#     p = argparse.ArgumentParser(
#         description="Generalized UMAP main-figure plot with optional dataset "
#                     "subsetting, cluster highlighting, SMILES-of-interest "
#                     "highlighting, and NP-class / sim-vs-real coloring.",
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter,
#     )
#     p.add_argument("--data-dir", type=Path, required=True)
#     p.add_argument("--prefix", type=str, default="cotrain")
#     p.add_argument("--splits", nargs="+", default=["train"], choices=["train", "val", "test"])
#     p.add_argument("--embedding-key", type=str, default="global_cond")

#     p.add_argument("--datasets", nargs="+", default=None,
#                    help="Restrict to these source datasets (e.g. spectranp "
#                         "nmrexp). Default: use everything loaded.")

#     p.add_argument("--cluster-labels-path", type=Path, default=None,
#                    help="cotrain_cluster_labs.npy from cluster_cotrain_molecules.py "
#                         "(required for --color-by cluster or --highlight-clusters).")
#     p.add_argument("--np-class-labels-path", type=Path, default=None,
#                    help="labels.pkl from npclassifier_local.py: List[str] of "
#                         "NPClassifier's dominant class per molecule, aligned to "
#                         "the unfiltered data (required for --color-by np_class).")
#     p.add_argument("--np-class-top-n", type=int, default=10,
#                    help="Number of most common NP classes to color distinctly; "
#                         "the rest are bucketed into a gray 'Other'.")
#     p.add_argument("--real-prefixes", nargs="+", default=["real"],
#                    help="Case-insensitive prefixes of `dataset` values counted as 'real' "
#                         "(for --color-by sim_real).")
#     p.add_argument("--sim-prefixes", nargs="+", default=["syn"],
#                    help="Case-insensitive prefixes of `dataset` values counted as "
#                         "'synthetic' (for --color-by sim_real).")

#     p.add_argument("--color-by", choices=["dataset", "cluster", "np_class", "sim_real", "none"],
#                    default="dataset")
#     p.add_argument("--highlight-clusters", nargs="+", type=int, default=None,
#                    help="Cluster IDs to rectangle-highlight + inset. Only "
#                         "drawn if explicitly given.")

#     p.add_argument("--smiles-of-interest", nargs="+", default=None,
#                    help="SMILES strings to star-highlight + inset (matched "
#                         "against the loaded data via canonical SMILES).")

#     p.add_argument("--umap-embedding-path", type=Path, default=None,
#                    help="Precomputed 2D projection (.npy), aligned to the "
#                         "exact subset being plotted. If omitted, UMAP is fit "
#                         "fresh on --embedding-key.")
#     p.add_argument("--save-umap-embedding", type=Path, default=None,
#                    help="Where to cache a freshly-fit 2D projection.")
#     p.add_argument("--umap-n-neighbors", type=int, default=30)
#     p.add_argument("--umap-min-dist", type=float, default=0.1)
#     p.add_argument("--umap-metric", type=str, default="cosine")

#     p.add_argument("--window-size", type=float, default=2.5)
#     p.add_argument("--n-mols-per-region", type=int, default=6)
#     p.add_argument("--mols-per-row", type=int, default=2)

#     p.add_argument("--out-dir", type=Path, default=Path("umap_figure"))
#     p.add_argument("--main-fig-name", type=str, default="umap_main.png")
#     p.add_argument("--dpi", type=int, default=600)
#     p.add_argument("--seed", type=int, default=0)
#     return p.parse_args(argv)


# def main(argv: Optional[Sequence[str]] = None) -> None:
#     global WINDOW_SIZE_GLOBAL

#     args = parse_args(argv)
#     WINDOW_SIZE_GLOBAL = args.window_size
#     args.out_dir.mkdir(parents=True, exist_ok=True)

#     if args.color_by == "cluster" and args.cluster_labels_path is None:
#         raise ValueError("--color-by cluster requires --cluster-labels-path.")
#     if args.highlight_clusters and args.cluster_labels_path is None:
#         raise ValueError("--highlight-clusters requires --cluster-labels-path.")
#     if args.color_by == "np_class" and args.np_class_labels_path is None:
#         raise ValueError("--color-by np_class requires --np-class-labels-path.")

#     print("=" * 78)
#     print("UMAP main figure")
#     print(f"  data dir           : {args.data_dir}")
#     print(f"  embedding key      : {args.embedding_key}")
#     print(f"  datasets           : {args.datasets or 'ALL'}")
#     print(f"  color-by           : {args.color_by}")
#     print(f"  highlight clusters : {args.highlight_clusters}")
#     print(f"  smiles of interest : {args.smiles_of_interest}")
#     print("=" * 78)

#     data = load_cotrain_data(args.data_dir, args.prefix, args.splits, args.embedding_key)
#     n_total = len(data["smiles"])

#     cluster_labels_full = None
#     if args.cluster_labels_path is not None:
#         cluster_labels_full = np.load(args.cluster_labels_path)
#         if len(cluster_labels_full) != n_total:
#             raise ValueError(
#                 f"{args.cluster_labels_path} has {len(cluster_labels_full)} labels "
#                 f"but the loaded (unfiltered) data has {n_total} molecules -- check "
#                 f"--splits matches what was used to generate the cluster labels."
#             )

#     np_class_labels_full = None
#     if args.np_class_labels_path is not None:
#         np_class_labels_full = load_pickle_labels(args.np_class_labels_path, n_total, "NP-class")

#     data, filtered_extras = filter_by_datasets(
#         data, args.datasets,
#         {"cluster": cluster_labels_full, "np_class": np_class_labels_full},
#     )
#     cluster_labels = filtered_extras["cluster"]
#     np_class_labels = filtered_extras["np_class"]

#     smiles = data["smiles"]
#     embedding = data["embedding"]
#     dataset_labels = data["dataset"]

#     emb2d = load_or_compute_umap(
#         embedding, args.umap_embedding_path, args.save_umap_embedding,
#         args.umap_n_neighbors, args.umap_min_dist, args.umap_metric, args.seed,
#     )

#     regions: List[Region] = []
#     if args.highlight_clusters:
#         regions.extend(build_cluster_regions(args.highlight_clusters, cluster_labels, emb2d))
#     if args.smiles_of_interest:
#         matches = match_smiles_of_interest(args.smiles_of_interest, smiles)
#         regions.extend(build_smiles_regions(matches, emb2d, args.window_size))

#     plot_main_scatter(
#         emb2d, args.color_by, dataset_labels, cluster_labels, np_class_labels,
#         args.highlight_clusters, args.np_class_top_n, args.real_prefixes, args.sim_prefixes,
#         regions, args.out_dir / args.main_fig_name, args.dpi,
#     )

#     if regions:
#         for region in regions:
#             plot_region_panel(
#                 region, emb2d, args.color_by, dataset_labels, cluster_labels, np_class_labels,
#                 args.highlight_clusters, args.np_class_top_n, args.real_prefixes, args.sim_prefixes,
#                 smiles, args.n_mols_per_region, args.mols_per_row, args.seed, args.out_dir, args.dpi,
#             )

#     manifest = {
#         "data_dir": str(args.data_dir), "prefix": args.prefix, "splits": args.splits,
#         "embedding_key": args.embedding_key,
#         "datasets": args.datasets, "color_by": args.color_by,
#         "highlight_clusters": args.highlight_clusters,
#         "smiles_of_interest": args.smiles_of_interest,
#         "n_molecules_plotted": len(smiles), "window_size": args.window_size,
#         "regions": [
#             {"kind": r.kind, "id": str(r.region_id), "label": r.label, "center": r.center}
#             for r in regions
#         ],
#     }
#     (args.out_dir / "umap_figure_manifest.json").write_text(json.dumps(manifest, indent=2))
#     print("Done.")


# if __name__ == "__main__":
#     main()

#############################################################################################################################
#############################################################################################################################
# #############################################################################################################################
# #############################################################################################################################
# #############################################################################################################################


# #!/usr/bin/env python
# """
# umap_main_figure.py
# ======================

# Generalized rewrite of the original "2a_pretty_plot" UMAP figure script.
# Preserves its exact visual style (rainbow cluster colors + gray "other",
# thin zoom rectangles with matching-color cluster-id labels, side-by-side
# scatter-zoom + molecule-grid inset panels) while adding:

#     1. Dataset subsetting -- plot only e.g. {spectranp} or {spectranp,
#        nmrexp} instead of always all three cotrain sources.
#     2. SMILES-of-interest highlighting -- give a list of SMILES strings and
#        the script finds each in the loaded data (via canonical-SMILES
#        match), marks it with a star on the main scatter, and saves a
#        zoom + molecule-grid inset centered on it (neighborhood molecules,
#        with the query molecule labeled "query" in its grid legend).
#     3. Cluster highlighting is now fully optional -- clusters are only
#        colored/gray-masked/inset if you pass `--highlight-clusters`,
#        instead of the original's hardcoded `clusters_to_show`.

# Background coloring of the full scatter (`--color-by`) is independent of
# the highlight overlays above, so you can e.g. color by source dataset while
# still drawing cluster-zoom insets on top, or vice versa.

# Inputs
# ------
#     - `<prefix>_<split>_global_cond.pt`  (extract_cotrain_embeddings.py)
#     - `cotrain_cluster_labs.npy`          (cluster_cotrain_molecules.py;
#                                             only needed for --color-by
#                                             cluster or --highlight-clusters)
#     - a precomputed 2D projection (--umap-embedding-path), OR the script
#       fits UMAP itself on the (possibly subsetted) embeddings and can cache
#       the result via --save-umap-embedding.

# Usage
# -----
#     # Reproduce the original figure: all 3 sources, 5 highlighted clusters
#     python umap_main_figure.py \\
#         --data-dir /scratch/gpfs/ZHONGE/jc4587/nmr_embs_cotrain \\
#         --cluster-labels-path cluster_summary/cotrain_cluster_labs.npy \\
#         --color-by cluster --highlight-clusters 0 48 99 149 199 \\
#         --out-dir umap_figure

#     # Only spectranp + nmrexp, colored by source, with 3 query molecules
#     python umap_main_figure.py \\
#         --data-dir /scratch/gpfs/ZHONGE/jc4587/nmr_embs_cotrain \\
#         --datasets spectranp nmrexp --color-by dataset \\
#         --smiles-of-interest "CCO" "c1ccccc1O" "CC(=O)Oc1ccccc1C(=O)O" \\
#         --out-dir umap_figure_subset
# """

# from __future__ import annotations

# import argparse
# import json
# import random
# from pathlib import Path
# from typing import Dict, List, Optional, Sequence, Tuple

# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap, BoundaryNorm
# from matplotlib.patches import Rectangle
# from matplotlib.gridspec import GridSpec

# from rdkit import Chem
# from rdkit.Chem import Draw
# from rdkit import RDLogger

# RDLogger.DisableLog("rdApp.*")

# QUALITATIVE_PALETTE = plt.cm.tab10(np.linspace(0, 1, 10))


# # -----------------------------------------------------------------------------
# # Data loading
# # -----------------------------------------------------------------------------


# def load_cotrain_data(data_dir: Path, prefix: str, splits: Sequence[str],
#                        embedding_key: str) -> Dict:
#     smiles: List[str] = []
#     dataset: List[str] = []
#     embed_parts: List[torch.Tensor] = []

#     for split in splits:
#         path = data_dir / f"{prefix}_{split}_global_cond.pt"
#         if not path.exists():
#             raise FileNotFoundError(f"Expected extraction output not found: {path}")
#         print(f"Loading {path} ...")
#         d = torch.load(path, map_location="cpu")
#         if embedding_key not in d:
#             raise KeyError(f"'{embedding_key}' not found in {path}. Keys: {list(d.keys())}")
#         smiles.extend(d["smiles"])
#         dataset.extend(d["dataset"])
#         embed_parts.append(d[embedding_key])

#     embedding = torch.cat(embed_parts, dim=0).numpy().astype(np.float32)
#     return {"smiles": smiles, "dataset": np.asarray(dataset), "embedding": embedding}


# def filter_by_datasets(data: Dict, datasets: Optional[Sequence[str]],
#                         cluster_labels: Optional[np.ndarray]
#                         ) -> Tuple[Dict, Optional[np.ndarray]]:
#     if datasets is None:
#         return data, cluster_labels

#     mask = np.isin(data["dataset"], list(datasets))
#     n_before, n_after = len(data["dataset"]), int(mask.sum())
#     if n_after == 0:
#         raise ValueError(f"No molecules found for --datasets {datasets}. "
#                           f"Available sources: {sorted(set(data['dataset'].tolist()))}")
#     print(f"Subsetting to datasets={list(datasets)}: {n_before} -> {n_after} molecules")

#     filtered = {
#         "smiles": [s for s, m in zip(data["smiles"], mask) if m],
#         "dataset": data["dataset"][mask],
#         "embedding": data["embedding"][mask],
#     }
#     filtered_labels = cluster_labels[mask] if cluster_labels is not None else None
#     return filtered, filtered_labels


# # -----------------------------------------------------------------------------
# # UMAP (load-or-compute, cacheable)
# # -----------------------------------------------------------------------------


# def load_or_compute_umap(embedding: np.ndarray, load_path: Optional[Path],
#                           save_path: Optional[Path], n_neighbors: int,
#                           min_dist: float, metric: str, seed: int) -> np.ndarray:
#     if load_path is not None:
#         print(f"Loading precomputed 2D projection from {load_path}")
#         emb2d = np.load(load_path)
#         if emb2d.shape[0] != embedding.shape[0]:
#             raise ValueError(
#                 f"{load_path} has {emb2d.shape[0]} rows but the (possibly "
#                 f"subsetted) loaded data has {embedding.shape[0]} molecules "
#                 f"-- a precomputed projection must be aligned to the exact "
#                 f"subset being plotted."
#             )
#         return emb2d

#     import umap.umap_ as umap_lib

#     print(f"Fitting UMAP (n_neighbors={n_neighbors}, min_dist={min_dist}, "
#           f"metric={metric}) on {embedding.shape[0]} points ...")
#     reducer = umap_lib.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
#                              metric=metric, random_state=seed)
#     emb2d = reducer.fit_transform(embedding)

#     if save_path is not None:
#         save_path.parent.mkdir(parents=True, exist_ok=True)
#         np.save(save_path, emb2d)
#         print(f"✓ Cached 2D projection to {save_path}")

#     return emb2d


# # -----------------------------------------------------------------------------
# # SMILES-of-interest matching
# # -----------------------------------------------------------------------------


# def canonicalize(smi: str) -> Optional[str]:
#     mol = Chem.MolFromSmiles(smi)
#     if mol is None:
#         return None
#     return Chem.MolToSmiles(mol)


# def match_smiles_of_interest(query_smiles: Sequence[str],
#                               dataset_smiles: List[str]) -> Dict[str, int]:
#     """Returns {query_smiles: index_in_dataset}, skipping (with a warning)
#     any query that doesn't canonicalize or isn't found in the dataset."""
#     canon_to_idx: Dict[str, int] = {}
#     for i, s in enumerate(dataset_smiles):
#         c = canonicalize(s)
#         if c is not None and c not in canon_to_idx:
#             canon_to_idx[c] = i

#     matches = {}
#     for q in query_smiles:
#         cq = canonicalize(q)
#         if cq is None:
#             print(f"[warn] Could not parse query SMILES, skipping: {q!r}")
#             continue
#         if cq not in canon_to_idx:
#             print(f"[warn] Query SMILES not found in the loaded/subsetted "
#                   f"data, skipping: {q!r}")
#             continue
#         matches[q] = canon_to_idx[cq]
#     return matches


# # -----------------------------------------------------------------------------
# # Region definitions (clusters and/or SMILES-of-interest)
# # -----------------------------------------------------------------------------


# class Region:
#     def __init__(self, kind: str, region_id, label: str, color, center: Tuple[float, float],
#                  member_idx: np.ndarray, query_idx: Optional[int] = None):
#         self.kind = kind                # "cluster" | "smiles"
#         self.region_id = region_id      # cluster int id, or the raw query smiles string
#         self.label = label              # text drawn on the main plot
#         self.color = color
#         self.center = center
#         self.member_idx = member_idx    # indices used to sample the molecule-grid panel
#         self.query_idx = query_idx      # for "smiles" regions: the exact matched index


# def build_cluster_regions(cluster_ids: Sequence[int], labels: np.ndarray,
#                            emb2d: np.ndarray) -> List[Region]:
#     n_normal = int(labels.max()) + 1
#     base_colors = plt.cm.rainbow(np.linspace(0, 1, n_normal))

#     regions = []
#     for cid in cluster_ids:
#         cluster_mask = labels == cid
#         pts = emb2d[cluster_mask]
#         if len(pts) == 0:
#             print(f"[warn] Cluster {cid} has no members in the current "
#                   f"(possibly subsetted) data, skipping.")
#             continue
#         center = (float(np.median(pts[:, 0])), float(np.median(pts[:, 1])))
#         member_idx = np.where(cluster_mask)[0]
#         color = base_colors[cid % n_normal]
#         regions.append(Region("cluster", cid, str(cid), color, center, member_idx))
#     return regions


# def build_smiles_regions(matches: Dict[str, int], emb2d: np.ndarray,
#                           window_size: float) -> List[Region]:
#     regions = []
#     for i, (query, idx) in enumerate(matches.items()):
#         center = (float(emb2d[idx, 0]), float(emb2d[idx, 1]))
#         half = window_size / 2
#         in_window = (
#             (emb2d[:, 0] >= center[0] - half) & (emb2d[:, 0] <= center[0] + half) &
#             (emb2d[:, 1] >= center[1] - half) & (emb2d[:, 1] <= center[1] + half)
#         )
#         member_idx = np.where(in_window)[0]
#         color = QUALITATIVE_PALETTE[i % len(QUALITATIVE_PALETTE)]
#         label = f"S{i+1}"
#         regions.append(Region("smiles", query, label, color, center, member_idx, query_idx=idx))
#     return regions


# # -----------------------------------------------------------------------------
# # Molecule sampling + rendering (same primitives as the original script)
# # -----------------------------------------------------------------------------


# def sample_region_smiles(region: Region, smiles: List[str], n_mols: int,
#                           seed: int) -> Tuple[List, List[str]]:
#     """Returns (rdkit_mols, legends). For 'smiles' regions the query molecule
#     is always included first and legended 'query'."""
#     rng = random.Random(seed + (region.query_idx if region.query_idx is not None else hash(region.region_id) % 10_000))

#     ordered_idx = list(region.member_idx)
#     rng.shuffle(ordered_idx)

#     chosen_idx = []
#     if region.kind == "smiles" and region.query_idx is not None:
#         chosen_idx.append(region.query_idx)

#     for idx in ordered_idx:
#         if len(chosen_idx) >= n_mols:
#             break
#         if idx in chosen_idx:
#             continue
#         if Chem.MolFromSmiles(smiles[idx]) is None:
#             continue
#         chosen_idx.append(idx)

#     mols, legends = [], []
#     for idx in chosen_idx:
#         mol = Chem.MolFromSmiles(smiles[idx])
#         if mol is None:
#             continue
#         mols.append(mol)
#         legends.append("query" if (region.kind == "smiles" and idx == region.query_idx) else "")
#     return mols, legends


# def mols_to_image(mols: List, legends: Optional[List[str]] = None,
#                    mols_per_row: int = 2, sub_img_size: Tuple[int, int] = (180, 180)):
#     if len(mols) == 0:
#         return None
#     return Draw.MolsToGridImage(
#         mols, molsPerRow=mols_per_row, subImgSize=sub_img_size,
#         legends=legends, returnPNG=False,
#     )


# # -----------------------------------------------------------------------------
# # Background coloring for the full scatter
# # -----------------------------------------------------------------------------


# def color_by_dataset(dataset_labels: np.ndarray) -> Tuple[np.ndarray, Dict[str, tuple]]:
#     sources = sorted(set(dataset_labels.tolist()))
#     palette = {src: QUALITATIVE_PALETTE[i % len(QUALITATIVE_PALETTE)] for i, src in enumerate(sources)}
#     colors = np.array([palette[s] for s in dataset_labels])
#     return colors, palette


# def color_by_cluster(labels: np.ndarray, highlight_clusters: Optional[Sequence[int]]
#                       ) -> Tuple[np.ndarray, ListedColormap, BoundaryNorm, np.ndarray]:
#     n_normal = int(labels.max()) + 1
#     base_colors = plt.cm.rainbow(np.linspace(0, 1, n_normal))
#     gray = np.array([[0.75, 0.75, 0.75, 1.0]])
#     custom_cmap = ListedColormap(np.vstack([base_colors, gray]))

#     labels_plot = labels.copy()
#     special_label = n_normal  # == labels.max() + 1

#     if highlight_clusters is not None:
#         mask = ~np.isin(labels_plot, list(highlight_clusters))
#         labels_plot[mask] = special_label

#     bounds = np.arange(special_label + 2) - 0.5
#     norm = BoundaryNorm(bounds, custom_cmap.N)
#     return labels_plot, custom_cmap, norm, base_colors


# # -----------------------------------------------------------------------------
# # Plotting
# # -----------------------------------------------------------------------------


# def plot_main_scatter(emb2d: np.ndarray, color_by: str, dataset_labels: np.ndarray,
#                        cluster_labels: Optional[np.ndarray], highlight_clusters: Optional[Sequence[int]],
#                        regions: List[Region], out_path: Path, dpi: int) -> None:
#     fig, ax = plt.subplots(figsize=(10, 10))

#     if color_by == "cluster":
#         labels_plot, cmap, norm, _ = color_by_cluster(cluster_labels, highlight_clusters)
#         ax.scatter(emb2d[:, 0], emb2d[:, 1], c=labels_plot, cmap=cmap, norm=norm,
#                    s=0.25, alpha=0.25, linewidths=0)
#     elif color_by == "dataset":
#         colors, palette = color_by_dataset(dataset_labels)
#         ax.scatter(emb2d[:, 0], emb2d[:, 1], c=colors, s=0.25, alpha=0.25, linewidths=0)
#         handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
#                                markersize=8, label=src) for src, c in palette.items()]
#         ax.legend(handles=handles, loc="best", frameon=True, fontsize=9)
#     else:
#         ax.scatter(emb2d[:, 0], emb2d[:, 1], c="steelblue", s=0.25, alpha=0.25, linewidths=0)

#     # Highlight overlays (rectangles + labels for clusters and/or SMILES regions)
#     for region in regions:
#         half = WINDOW_SIZE_GLOBAL / 2
#         rect = Rectangle(
#             (region.center[0] - half, region.center[1] - half),
#             WINDOW_SIZE_GLOBAL, WINDOW_SIZE_GLOBAL,
#             fill=False, linewidth=2, edgecolor=region.color,
#         )
#         ax.add_patch(rect)

#         marker = "*" if region.kind == "smiles" else None
#         if marker:
#             ax.scatter([region.center[0]], [region.center[1]], marker=marker, s=120,
#                        facecolor=region.color, edgecolor="black", linewidths=0.5, zorder=6)

#         ax.text(region.center[0], region.center[1] + half + 0.2, region.label,
#                 color=region.color, ha="center", fontsize=10, weight="bold")

#     ax.set_xlabel("UMAP-1")
#     ax.set_ylabel("UMAP-2")
#     plt.tight_layout()
#     plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
#     plt.close(fig)
#     print(f"✓ Saved {out_path}")


# def plot_region_panel(region: Region, emb2d: np.ndarray, color_by: str,
#                        dataset_labels: np.ndarray, cluster_labels: Optional[np.ndarray],
#                        highlight_clusters: Optional[Sequence[int]], smiles: List[str],
#                        n_mols: int, mols_per_row: int, seed: int, out_dir: Path, dpi: int) -> None:
#     fig = plt.figure(figsize=(8, 4))
#     gs = GridSpec(1, 2, width_ratios=[1, 1], figure=fig)

#     ax_scatter = fig.add_subplot(gs[0, 0])

#     if color_by == "cluster":
#         labels_plot, cmap, norm, _ = color_by_cluster(cluster_labels, highlight_clusters)
#         ax_scatter.scatter(emb2d[:, 0], emb2d[:, 1], c=labels_plot, cmap=cmap, norm=norm,
#                            s=1, alpha=0.8, linewidths=0)
#     elif color_by == "dataset":
#         colors, _ = color_by_dataset(dataset_labels)
#         ax_scatter.scatter(emb2d[:, 0], emb2d[:, 1], c=colors, s=1, alpha=0.8, linewidths=0)
#     else:
#         ax_scatter.scatter(emb2d[:, 0], emb2d[:, 1], c="steelblue", s=1, alpha=0.8, linewidths=0)

#     half = WINDOW_SIZE_GLOBAL / 2
#     ax_scatter.set_xlim(region.center[0] - half, region.center[0] + half)
#     ax_scatter.set_ylim(region.center[1] - half, region.center[1] + half)
#     ax_scatter.set_aspect("equal")
#     ax_scatter.set_xticks([])
#     ax_scatter.set_yticks([])
#     title = f"Cluster {region.region_id}" if region.kind == "cluster" else f"{region.label}: query molecule"
#     ax_scatter.set_title(title)

#     mols, legends = sample_region_smiles(region, smiles, n_mols, seed)
#     mol_img = mols_to_image(mols, legends, mols_per_row=mols_per_row)

#     ax_mols = fig.add_subplot(gs[0, 1])
#     if mol_img is not None:
#         ax_mols.imshow(mol_img)
#     else:
#         ax_mols.text(0.5, 0.5, "no valid SMILES\nsampled", ha="center", va="center",
#                      transform=ax_mols.transAxes)
#     ax_mols.set_xticks([])
#     ax_mols.set_yticks([])
#     ax_mols.set_title("sampled molecules")

#     plt.tight_layout()

#     region_tag = region.region_id if region.kind == "cluster" else region.label
#     out_path = out_dir / f"region_{region.kind}_{region_tag}.png"
#     plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
#     plt.close(fig)

#     if mol_img is not None:
#         mol_img.save(out_dir / f"region_{region.kind}_{region_tag}_mols.png")

#     print(f"✓ Saved {out_path}")


# # -----------------------------------------------------------------------------
# # CLI
# # -----------------------------------------------------------------------------

# WINDOW_SIZE_GLOBAL = 2.5  # set from --window-size in main(); module-level for plot fns


# def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
#     p = argparse.ArgumentParser(
#         description="Generalized UMAP main-figure plot with optional dataset "
#                      "subsetting, cluster highlighting, and SMILES-of-interest "
#                      "highlighting.",
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter,
#     )
#     p.add_argument("--data-dir", type=Path, required=True)
#     p.add_argument("--prefix", type=str, default="cotrain")
#     p.add_argument("--splits", nargs="+", default=["train"], choices=["train", "val", "test"])
#     p.add_argument("--embedding-key", type=str, default="global_cond")

#     p.add_argument("--datasets", nargs="+", default=None,
#                    help="Restrict to these source datasets (e.g. spectranp "
#                         "nmrexp). Default: use everything loaded.")

#     p.add_argument("--cluster-labels-path", type=Path, default=None,
#                    help="cotrain_cluster_labs.npy from cluster_cotrain_molecules.py "
#                         "(required for --color-by cluster or --highlight-clusters).")
#     p.add_argument("--color-by", choices=["dataset", "cluster", "none"], default="dataset")
#     p.add_argument("--highlight-clusters", nargs="+", type=int, default=None,
#                    help="Cluster IDs to rectangle-highlight + inset. Only "
#                         "drawn if explicitly given.")

#     p.add_argument("--smiles-of-interest", nargs="+", default=None,
#                    help="SMILES strings to star-highlight + inset (matched "
#                         "against the loaded data via canonical SMILES).")

#     p.add_argument("--umap-embedding-path", type=Path, default=None,
#                    help="Precomputed 2D projection (.npy), aligned to the "
#                         "exact subset being plotted. If omitted, UMAP is fit "
#                         "fresh on --embedding-key.")
#     p.add_argument("--save-umap-embedding", type=Path, default=None,
#                    help="Where to cache a freshly-fit 2D projection.")
#     p.add_argument("--umap-n-neighbors", type=int, default=30)
#     p.add_argument("--umap-min-dist", type=float, default=0.1)
#     p.add_argument("--umap-metric", type=str, default="cosine")

#     p.add_argument("--window-size", type=float, default=2.5)
#     p.add_argument("--n-mols-per-region", type=int, default=6)
#     p.add_argument("--mols-per-row", type=int, default=2)

#     p.add_argument("--out-dir", type=Path, default=Path("umap_figure"))
#     p.add_argument("--main-fig-name", type=str, default="umap_main.png")
#     p.add_argument("--dpi", type=int, default=600)
#     p.add_argument("--seed", type=int, default=0)
#     return p.parse_args(argv)


# def main(argv: Optional[Sequence[str]] = None) -> None:
#     global WINDOW_SIZE_GLOBAL

#     args = parse_args(argv)
#     WINDOW_SIZE_GLOBAL = args.window_size
#     args.out_dir.mkdir(parents=True, exist_ok=True)

#     if args.color_by == "cluster" and args.cluster_labels_path is None:
#         raise ValueError("--color-by cluster requires --cluster-labels-path.")
#     if args.highlight_clusters and args.cluster_labels_path is None:
#         raise ValueError("--highlight-clusters requires --cluster-labels-path.")

#     print("=" * 78)
#     print("UMAP main figure")
#     print(f"  data dir           : {args.data_dir}")
#     print(f"  datasets           : {args.datasets or 'ALL'}")
#     print(f"  color-by           : {args.color_by}")
#     print(f"  highlight clusters : {args.highlight_clusters}")
#     print(f"  smiles of interest : {args.smiles_of_interest}")
#     print("=" * 78)

#     data = load_cotrain_data(args.data_dir, args.prefix, args.splits, args.embedding_key)

#     cluster_labels_full = None
#     if args.cluster_labels_path is not None:
#         cluster_labels_full = np.load(args.cluster_labels_path)
#         if len(cluster_labels_full) != len(data["smiles"]):
#             raise ValueError(
#                 f"{args.cluster_labels_path} has {len(cluster_labels_full)} labels "
#                 f"but the loaded (unfiltered) data has {len(data['smiles'])} "
#                 f"molecules -- check --splits matches what was used to "
#                 f"generate the cluster labels."
#             )

#     data, cluster_labels = filter_by_datasets(data, args.datasets, cluster_labels_full)
#     smiles = data["smiles"]
#     embedding = data["embedding"]
#     dataset_labels = data["dataset"]

#     emb2d = load_or_compute_umap(
#         embedding, args.umap_embedding_path, args.save_umap_embedding,
#         args.umap_n_neighbors, args.umap_min_dist, args.umap_metric, args.seed,
#     )

#     regions: List[Region] = []
#     if args.highlight_clusters:
#         regions.extend(build_cluster_regions(args.highlight_clusters, cluster_labels, emb2d))
#     if args.smiles_of_interest:
#         matches = match_smiles_of_interest(args.smiles_of_interest, smiles)
#         regions.extend(build_smiles_regions(matches, emb2d, args.window_size))

#     plot_main_scatter(
#         emb2d, args.color_by, dataset_labels, cluster_labels, args.highlight_clusters,
#         regions, args.out_dir / args.main_fig_name, args.dpi,
#     )

#     if regions:
#         for region in regions:
#             plot_region_panel(
#                 region, emb2d, args.color_by, dataset_labels, cluster_labels,
#                 args.highlight_clusters, smiles, args.n_mols_per_region,
#                 args.mols_per_row, args.seed, args.out_dir, args.dpi,
#             )

#     manifest = {
#         "data_dir": str(args.data_dir), "prefix": args.prefix, "splits": args.splits,
#         "datasets": args.datasets, "color_by": args.color_by,
#         "highlight_clusters": args.highlight_clusters,
#         "smiles_of_interest": args.smiles_of_interest,
#         "n_molecules_plotted": len(smiles), "window_size": args.window_size,
#         "regions": [
#             {"kind": r.kind, "id": str(r.region_id), "label": r.label, "center": r.center}
#             for r in regions
#         ],
#     }
#     (args.out_dir / "umap_figure_manifest.json").write_text(json.dumps(manifest, indent=2))
#     print("Done.")


# if __name__ == "__main__":
#     main()
