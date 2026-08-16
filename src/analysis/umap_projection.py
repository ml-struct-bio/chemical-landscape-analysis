"""UMAP projection of an embedding, plus every column a figure might colour by.

The split this module enforces: **the projection and the colour columns are
analysis; which of them to draw is plotting.** One run stores the 2-D coordinates
alongside the dataset labels, split tags, Butina cluster ids, real/synthetic
classification, molecular properties and NMR spectral features -- all aligned to
the same rows -- and the plotting script then emits as many figures as asked
without ever refitting.

That is why the previous pipeline needed three scripts (`5`, `6`, `7`) and a
batch driver that re-ran the whole thing once per colouring.

Computation only -- never imports matplotlib.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.analysis.anchors import build_anchor_artifacts
from src.analysis.umap_cache import (
    DEFAULT_METRIC_BY_EMBEDDING,
    DEFAULT_SCALE_BY_EMBEDDING,
    resolve_umap,
)
from src.common.manifest import write_manifest


SCHEMA_VERSION = 1


def classify_sim_real(dataset_labels: np.ndarray, real_prefixes: Sequence[str],
                      sim_prefixes: Sequence[str]) -> np.ndarray:
    """real / synthetic / unknown, by prefix match on the `dataset` field."""
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


def encode_categorical(values: np.ndarray) -> tuple[np.ndarray, List[str]]:
    """(int codes, level names). Categoricals are stored as codes + a level list
    rather than as string arrays, so a 2.5M-row column costs 10 MB not 200."""
    levels = sorted({str(v) for v in values})
    lookup = {name: i for i, name in enumerate(levels)}
    return np.asarray([lookup[str(v)] for v in values], dtype=np.int32), levels


def load_cluster_labels(cluster_dir: Optional[Path], n_rows: int) -> Optional[np.ndarray]:
    """Butina labels from `data/03_clustering/<tag>/cluster_labels.npy`.

    Both are written in the extraction's row order, so alignment is positional --
    but only if the clustering was run over the SAME splits. A length mismatch
    means it was not, and colouring by it would attach every molecule to some
    other molecule's cluster, so it is refused rather than truncated.
    """
    if cluster_dir is None:
        return None
    path = Path(cluster_dir) / "cluster_labels.npy"
    if not path.exists():
        print(f"[note] No cluster labels at {path} -- 'cluster' colouring will be "
              f"unavailable. Run analysis/03_clustering.py to add it.")
        return None
    labels = np.load(path)
    if len(labels) != n_rows:
        raise SystemExit(
            f"Cluster labels at {path} cover {len(labels)} molecules but this corpus has "
            f"{n_rows}.\nThey are aligned positionally, so this means the clustering was "
            f"run over different --splits. Re-run analysis/03_clustering.py with the same "
            f"splits, or pass --cluster-tag pointing at one that matches.")
    print(f"  cluster labels: {len(np.unique(labels))} clusters from {path.parent.name}")
    return labels.astype(np.int32)


def project_and_store(
    *,
    embedding: np.ndarray,
    smiles: Sequence[str],
    dataset: np.ndarray,
    split: np.ndarray,
    properties: np.ndarray,
    property_names: Sequence[str],
    spectral: np.ndarray,
    spectral_names: Sequence[str],
    cluster_labels: Optional[np.ndarray],
    anchor_specs: Sequence = (),
    k_neighbors: int = 0,
    inset_clusters: Sequence[int] = (),
    n_region_mols: int = 6,
    seed: int = 1234,
    n_workers: int = 1,
    out_dir: Path,
    tag: str,
    slug: str,
    embedding_key: str,
    embedding_label: str,
    n_neighbors: int,
    min_dist: float,
    metric: Optional[str],
    pca_dim: Optional[int],
    scale: Optional[bool],
    umap_seed: int,
    cache_dir: Optional[Path],
    read_only_cache_dirs: Sequence[Optional[Path]] = (),
    datasets_filter: Optional[Sequence[str]] = None,
    real_prefixes: Sequence[str] = ("real",),
    sim_prefixes: Sequence[str] = ("syn",),
    params: Optional[Dict[str, Any]] = None,
    inputs: Sequence[Path] = (),
) -> Dict[str, Path]:
    """Projects `embedding` to 2-D and writes every colour column beside it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}

    metric = metric or DEFAULT_METRIC_BY_EMBEDDING.get(embedding_key, "cosine")
    if scale is None:
        scale = DEFAULT_SCALE_BY_EMBEDDING.get(embedding_key, True)

    # Exactly the fields the previous pipeline fingerprinted, in the same order
    # and types. Deviating here would miss every existing cached fit -- including
    # the full-corpus global_cond and ecfp projections, which cost hours.
    fingerprint_fields = {
        "embedding_key": embedding_key,
        "datasets": sorted(datasets_filter) if datasets_filter else None,
        "n_neighbors": n_neighbors,
        "min_dist": min_dist,
        "metric": metric,
        "pca_dim": pca_dim,
        "scale": scale,
        "seed": umap_seed,
        "n_points": int(embedding.shape[0]),
        "n_dims": int(embedding.shape[1]),
    }

    coords, _ = resolve_umap(
        embedding, fingerprint_fields, n_neighbors=n_neighbors, min_dist=min_dist,
        metric=metric, seed=umap_seed, pca_dim=pca_dim, scale=scale,
        cache_dir=cache_dir, read_only_cache_dirs=read_only_cache_dirs,
        need_bundle=False,
    )
    coords = np.asarray(coords, dtype=np.float32)
    if coords.shape[0] != embedding.shape[0]:
        raise SystemExit(
            f"UMAP returned {coords.shape[0]} points for a {embedding.shape[0]}-row "
            f"embedding. A cached fit was matched to the wrong point set -- the cache "
            f"fingerprint is a CONFIG hash, not a content hash, so a different corpus of "
            f"the same shape can collide. Use a separate --umap-cache-dir.")

    written["coords"] = out_dir / "coords.npz"
    np.savez_compressed(written["coords"], coords=coords)

    # --- categorical colour columns ------------------------------------------
    dataset_codes, dataset_levels = encode_categorical(dataset)
    split_codes, split_levels = encode_categorical(split)
    sim_real_codes, sim_real_levels = encode_categorical(
        classify_sim_real(dataset, real_prefixes, sim_prefixes))

    categorical: Dict[str, np.ndarray] = {
        "dataset_codes": dataset_codes,
        "dataset_levels": np.asarray(dataset_levels),
        "split_codes": split_codes,
        "split_levels": np.asarray(split_levels),
        "sim_real_codes": sim_real_codes,
        "sim_real_levels": np.asarray(sim_real_levels),
    }
    if cluster_labels is not None:
        categorical["cluster"] = cluster_labels
    written["categorical"] = out_dir / "categorical.npz"
    np.savez_compressed(written["categorical"], **categorical)

    # --- continuous colour columns -------------------------------------------
    written["properties"] = out_dir / "properties.npz"
    np.savez_compressed(written["properties"], values=properties.astype(np.float32),
                        names=np.asarray(list(property_names)))
    written["spectral"] = out_dir / "spectral.npz"
    np.savez_compressed(written["spectral"], values=np.asarray(spectral, dtype=np.float32),
                        names=np.asarray(list(spectral_names)))

    # --- anchors, neighbours, inset regions ----------------------------------
    # All of it is bounded: a handful of anchors, k neighbours each, and a
    # capped sample of molecules per region. That is why SMILES can be stored
    # here at all -- keeping the corpus-wide list would be ~120 MB of strings
    # that no figure reads.
    anchor_tables = build_anchor_artifacts(
        embedding=embedding, coords=coords, smiles=smiles,
        anchor_specs=anchor_specs, k_neighbors=k_neighbors,
        cluster_labels=cluster_labels, inset_clusters=inset_clusters,
        n_region_mols=n_region_mols, seed=seed, n_workers=n_workers,
    )
    for name, frame in anchor_tables.items():
        if len(frame):
            written[name] = out_dir / f"{name}.csv"
            frame.to_csv(written[name], index=False)
    n_matched = int((anchor_tables["anchors"]["row"] >= 0).sum()) \
        if len(anchor_tables["anchors"]) else 0

    write_manifest(
        out_dir, slug=slug, tag=tag, schema_version=SCHEMA_VERSION,
        params={
            **(params or {}),
            "embedding": embedding_label,
            "embedding_key": embedding_key,
            "embedding_dim": int(embedding.shape[1]),
            "n_molecules": int(embedding.shape[0]),
            "umap": {"n_neighbors": n_neighbors, "min_dist": min_dist, "metric": metric,
                     "pca_dim": pca_dim, "scale": bool(scale), "seed": umap_seed},
            "umap_fingerprint_fields": fingerprint_fields,
            "property_names": list(property_names),
            "spectral_names": list(spectral_names),
            "dataset_levels": dataset_levels,
            "split_levels": split_levels,
            "sim_real_levels": sim_real_levels,
            "n_clusters": (int(len(np.unique(cluster_labels)))
                           if cluster_labels is not None else None),
            "real_prefixes": list(real_prefixes),
            "sim_prefixes": list(sim_prefixes),
            "anchors": [str(a) for a, _ in anchor_specs],
            "n_anchors_matched": n_matched,
            "k_neighbors": int(k_neighbors),
            "neighbor_metric": "cosine",
            "inset_clusters": [int(c) for c in inset_clusters],
            "n_region_mols": int(n_region_mols),
            "n_regions": int(len(anchor_tables["regions"])),
        },
        inputs=list(inputs),
        outputs=list(written.values()),
    )
    return written
