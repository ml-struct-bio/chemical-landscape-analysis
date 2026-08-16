"""Decoder layer-comparison analysis.

Port of the previous pipeline's `src/analysis/layer_comparison_analysis.py`
(`10_run_layer_comparison_experiment.py`). Compares decoder per-layer hidden-
state representations against each other and against two baselines -- ECFP and
the peak embedder's `global_cond` -- along three axes: property linear-
decodability, unsupervised cluster-quality vs. Butina labels, and ECFP-vs-
embedding nearest-neighbor agreement. Also resolves PC-traversal filmstrip
steps (real molecules only, via `src/analysis/geometry.py`) for a handful of
representative decoder layers.

Differences from the previous pipeline's port, all deliberate:

  - Loading and alignment go through `src/analysis/corpus.py`'s
    `load_decoder_corpus`/`align_to_corpus` instead of a hand-rolled
    `align_indices` -- the same join `04_joint_pca`/`05_umap` use, which tries
    raw-SMILES equality before falling back to (parallel) canonical-SMILES
    matching only for what's left, rather than canonicalizing the entire
    corpus single-threaded up front.
  - No property-direction traversal (the old `fit_property_direction` /
    `property_traversal_plot`) -- PC traversal only, reusing
    `geometry.traversal_steps` exactly as `04_joint_pca` does.
  - No per-representation PC1/PC2-vs-descriptor scatter grid (the old
    `plot_pc_property_correlation_grid`, one PNG per representation, ~38 on a
    full sweep). The PC-interpretability SWEEP chart plus the full
    `pc_correlations.csv` numbers are kept; the redundant scatter grids are
    not.
  - Loading + aligning every (layer, timestep) of one stream is by far the
    most expensive step here (tens of GB read + a corpus-wide join), and has
    nothing to do with which metrics or traversal layers are requested. It is
    cached under `cache/layer_comparison/`, keyed on the load configuration
    (NOT a content hash -- see `_load_aligned_representations`), so re-running
    with different `--traversal-layers`/`--n-steps`/percentiles reuses it.
    `--refit` bypasses the cache.
  - No `--plot-only`: this analysis writes CSV/NPZ only (no pickled
    artifact), and replotting is just re-running `plotting/07_layer_comparison.py`.

Computation only -- never imports matplotlib.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.analysis.corpus import EmbeddingSpec, available_decoder_keys, cached_descriptor_matrix, load_decoder_corpus
from src.analysis.descriptors import DEFAULT_DESCRIPTOR_NAMES, best_per_pc, compute_pc_correlations
from src.analysis.geometry import traversal_steps
from src.analysis.layer_comparison_metrics import (
    PROPERTY_FUNCS,
    compute_cluster_quality,
    compute_nn_overlap,
    compute_property_matrix,
    compute_property_r2,
    pairwise_tanimoto,
    stratified_subsample_indices,
)
from src.common.manifest import write_manifest
from src.common.paths import cache_dir
from src.common.paths import data_dir as tag_data_dir

SCHEMA_VERSION = 1

# Baselines sit left of the decoder layers on the x-axis, at fixed sentinel
# positions -- meaningless as coordinates, just an ordering.
BASELINE_X = {"ECFP": -3.0, "global_cond": -1.5}


# -----------------------------------------------------------------------------
# --traversal-layers parsing (validated before any loading -- see
# validate_traversal_keys)
# -----------------------------------------------------------------------------


def _parse_traversal_key(s: str) -> Tuple[int, Optional[float]]:
    """Parses one --traversal-layers entry into (layer, timestep).

    Accepts both `'6,0.001'` (explicit timestep) and a bare `'6'`, which
    returns timestep None meaning "the first timestep available for that
    layer" -- resolved later, once the layerwise file has been read and the
    available timesteps are actually known.
    """
    parts = [p.strip() for p in s.split(",")]
    if len(parts) == 1:
        layer_str, ts_str = parts[0], None
    elif len(parts) == 2:
        layer_str, ts_str = parts
    else:
        raise ValueError(
            f"--traversal-layers entry {s!r} has {len(parts)} comma-separated fields; "
            f"expected 'LAYER' or 'LAYER,TIMESTEP' (e.g. '6' or '6,0.001')."
        )
    try:
        layer = int(layer_str)
    except ValueError:
        raise ValueError(f"--traversal-layers entry {s!r}: layer {layer_str!r} is not an integer.") from None
    if ts_str is None or ts_str == "":
        return layer, None
    try:
        return layer, float(ts_str)
    except ValueError:
        raise ValueError(f"--traversal-layers entry {s!r}: timestep {ts_str!r} is not a number.") from None


def validate_traversal_keys(traversal_layers: Optional[Sequence[str]]) -> None:
    """Fail-fast check for --traversal-layers, safe to call before any data is
    loaded. The real resolution against available (layer, timestep) keys still
    happens later; this only catches malformed strings."""
    for s in traversal_layers or []:
        _parse_traversal_key(s)


# -----------------------------------------------------------------------------
# Loading + alignment (cached -- see module docstring)
# -----------------------------------------------------------------------------


def _cache_fingerprint(layerwise_dir: Path, data_dir: Path, prefix: str,
                       splits: Sequence[str], stream: str) -> str:
    payload = {"layerwise_dir": str(layerwise_dir), "data_dir": str(data_dir),
               "prefix": prefix, "splits": list(splits), "stream": stream}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:32]


def _slug_for(stream: str, layer: int, timestep: float) -> str:
    return EmbeddingSpec(kind="decoder", stream=stream, layer=layer, timestep=timestep).slug


def _load_aligned_representations(layerwise_dir: Path, data_dir: Path, prefix: str,
                                  splits: Sequence[str], stream: str, n_workers: int,
                                  refit: bool) -> Dict[str, Any]:
    """Every (layer, timestep) representation of `stream`, aligned to the
    ECFP/global_cond baselines and (via `peak_index`) to the full encoder
    corpus's row order for cluster-label lookup.

    Cached under `cache/layer_comparison/<fingerprint>.pkl`. The fingerprint is
    a CONFIG hash (directories/prefix/splits/stream), not a content hash --
    swapping in different data under the same configuration will
    false-positive match a stale entry, same caveat as the UMAP cache.
    `refit=True` bypasses it.
    """
    fp = _cache_fingerprint(layerwise_dir, data_dir, prefix, splits, stream)
    path = cache_dir("layer_comparison", create=True) / f"{fp}.pkl"
    if path.exists() and not refit:
        print(f"Reusing cached aligned representations {path.name} (pass --refit to redo it).")
        with open(path, "rb") as f:
            return pickle.load(f)

    keys = [k for k in available_decoder_keys(layerwise_dir, prefix, splits) if k[0] == stream]
    if not keys:
        raise SystemExit(f"No (layer, timestep) combinations for stream {stream!r} in the "
                         f"layerwise file at {layerwise_dir}.")
    specs = [EmbeddingSpec(kind="decoder", stream=s, layer=l, timestep=t) for s, l, t in keys]
    layer_keys = sorted({(l, t) for _s, l, t in keys}, key=lambda lt: (lt[1], lt[0]))

    corpus = load_decoder_corpus(data_dir, layerwise_dir, data_dir, prefix, splits, specs,
                                 n_workers=n_workers, require_spectral=False,
                                 embedding_keys=("global_cond", "ecfp"))

    result = {
        "smiles": corpus["smiles"],
        "dataset": corpus["dataset"],
        "ecfp": corpus["baseline_embeddings"]["ecfp"],
        "global_cond": corpus["baseline_embeddings"]["global_cond"],
        "decoder": {spec.slug: corpus["embeddings"][(spec.stream, spec.layer, spec.timestep)]
                    for spec in specs},
        "layer_keys": layer_keys,
        "peak_index": corpus["peak_index"],
    }
    with open(path, "wb") as f:
        pickle.dump(result, f)
    print(f"Cached aligned representations -> {path}")
    return result


def _load_cluster_labels(cluster_tag: Optional[str], peak_index: np.ndarray) -> Optional[np.ndarray]:
    """Butina labels gathered through the SAME join the decoder embeddings came
    through -- cluster_labels.npy is aligned to the full encoder corpus's row
    order, and the decoder molecule set is a joined subset of it in a
    different order (mirrors 05_umap.py's decoder branch)."""
    if not cluster_tag:
        return None
    path = tag_data_dir("03_clustering", cluster_tag) / "cluster_labels.npy"
    if not path.exists():
        print(f"[note] No cluster labels at {path} -- cluster-quality metrics skipped.")
        return None
    all_labels = np.load(path)
    if len(all_labels) <= int(peak_index.max()):
        print(f"[warn] cluster labels cover {len(all_labels)} molecules but the join reaches "
              f"row {int(peak_index.max())} -- the clustering was run over different --splits. "
              f"Skipping cluster-quality metrics.")
        return None
    labels = all_labels[peak_index].astype(np.int32)
    print(f"  cluster labels gathered through the decoder join ({len(np.unique(labels))} clusters)")
    return labels


# -----------------------------------------------------------------------------
# Summary text
# -----------------------------------------------------------------------------


def _summary_report(df: pd.DataFrame, property_names: List[str], include_silhouette: bool) -> str:
    lines = ["Decoder layer comparison: best configuration per metric", "=" * 72, ""]
    metrics_higher_better = [f"r2_{p}" for p in property_names] + [
        "calinski_harabasz", "knn_purity", "tanimoto_cosine_pearson", "tanimoto_cosine_spearman",
    ] + (["silhouette"] if include_silhouette else [])
    metrics_lower_better = ["davies_bouldin"]

    for m in metrics_higher_better:
        if m not in df.columns or df[m].isna().all():
            continue
        row = df.loc[df[m].idxmax()]
        lines.append(f"{m:28s}: best = {row['representation']:24s} ({m}={row[m]:.4f})")
    for m in metrics_lower_better:
        if m not in df.columns or df[m].isna().all():
            continue
        row = df.loc[df[m].idxmin()]
        lines.append(f"{m:28s}: best = {row['representation']:24s} ({m}={row[m]:.4f})")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Main orchestration
# -----------------------------------------------------------------------------


def run_layer_comparison(
    *,
    data_dir: Path,
    layerwise_dir: Path,
    prefix: str,
    splits: Sequence[str],
    out_dir: Path,
    tag: str,
    slug: str,
    stream: str = "x_hidden_mean",
    cluster_tag: Optional[str] = None,
    include_silhouette: bool = False,
    knn_purity_k: int = 25,
    nn_overlap_sample: int = 3000,
    k_neighbors: Sequence[int] = (5, 10, 25),
    n_prop_workers: int = 1,
    descriptor_names: Optional[Sequence[str]] = None,
    n_desc_workers: int = 1,
    traversal_layers: Optional[Sequence[str]] = None,
    n_pcs_traversal: int = 2,
    n_steps: int = 8,
    pc_pct_lo: float = 1.0,
    pc_pct_hi: float = 99.0,
    n_background_scatter: int = 20_000,
    device: str = "cpu",
    seed: int = 1234,
    refit: bool = False,
) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}
    rng = np.random.default_rng(seed)

    print("=" * 78)
    print("Decoder layer comparison")
    print(f"  layerwise dir    : {layerwise_dir}")
    print(f"  data dir         : {data_dir}")
    print(f"  stream           : {stream}")
    print(f"  cluster tag      : {cluster_tag or '(none -- cluster-quality metrics skipped)'}")
    print(f"  tag              : {tag}")
    print("=" * 78)

    aligned = _load_aligned_representations(layerwise_dir, data_dir, prefix, splits, stream,
                                            n_prop_workers, refit)
    smiles = aligned["smiles"]
    n_total = len(smiles)
    print(f"Aligned molecule set: {n_total} molecules")

    cluster_labels = _load_cluster_labels(cluster_tag, aligned["peak_index"])

    # ---- build the representation list: ECFP, global_cond, every decoder layer ----
    representations: Dict[str, np.ndarray] = {"ECFP": aligned["ecfp"], "global_cond": aligned["global_cond"]}
    layer_of: Dict[str, int] = {}
    for (layer, timestep) in aligned["layer_keys"]:
        slug_name = _slug_for(stream, layer, timestep)
        representations[slug_name] = aligned["decoder"][slug_name]
        layer_of[slug_name] = layer
    print(f"Representations to compare: {list(representations.keys())}")

    # ---- ECFP-vs-embedding NN-overlap subsample (shared Tanimoto matrix) ----
    nn_idx = stratified_subsample_indices(aligned["dataset"], nn_overlap_sample, rng)
    tanimoto = pairwise_tanimoto(aligned["ecfp"][nn_idx], device)

    # ---- properties + descriptors (computed once, reused across every representation) ----
    property_names = list(PROPERTY_FUNCS.keys())
    property_matrix = compute_property_matrix(smiles, n_prop_workers)

    descriptor_names = list(descriptor_names) if descriptor_names else list(DEFAULT_DESCRIPTOR_NAMES)
    descriptor_matrix = cached_descriptor_matrix(smiles, descriptor_names, n_desc_workers)

    # ---- per-representation metrics ----
    rows: List[Dict[str, Any]] = []
    pc_corr_rows: List[Dict[str, Any]] = []
    pc_best_rows: List[Dict[str, Any]] = []
    for name, x in representations.items():
        print(f"\n### {name} ###")
        x_scaled = StandardScaler().fit_transform(x).astype(np.float32)
        x_pos = BASELINE_X.get(name, float(layer_of.get(name, 0)))
        row: Dict[str, Any] = {"representation": name, "x_pos": x_pos}

        r2s = compute_property_r2(x_scaled, property_matrix, property_names, seed)
        row.update({f"r2_{k}": v for k, v in r2s.items()})

        if cluster_labels is not None:
            cq = compute_cluster_quality(x_scaled, cluster_labels, knn_purity_k, include_silhouette, seed)
            row.update(cq)

        nn = compute_nn_overlap(x[nn_idx], tanimoto, list(k_neighbors), device)
        row.update(nn)

        # PC1/PC2 vs. RDKit-descriptor correlations, same machinery
        # 04_joint_pca uses, restricted to the first two PCs and applied to
        # every representation so interpretability can be tracked across
        # decoder depth (and against the ECFP/global_cond baselines).
        pcs2 = PCA(n_components=2, random_state=seed).fit_transform(x_scaled)
        pc_corr_df = compute_pc_correlations(pcs2, descriptor_matrix, descriptor_names,
                                             desc=f"{name}: PC-descriptor correlations")
        pc_best_df = best_per_pc(pc_corr_df, column="descriptor")
        for pc in pc_corr_df.index:
            for desc in pc_corr_df.columns:
                pc_corr_rows.append({"representation": name, "x_pos": x_pos, "pc": pc,
                                     "descriptor": desc, "r": pc_corr_df.loc[pc, desc]})
        for pc_i, pc_label in enumerate(["pc1", "pc2"]):
            best_row = pc_best_df.iloc[pc_i]
            row[f"{pc_label}_best_descriptor"] = best_row["descriptor"]
            row[f"{pc_label}_best_r"] = best_row["r"]
            pc_best_rows.append({"representation": name, "x_pos": x_pos, "pc": best_row["pc"],
                                 "best_descriptor": best_row["descriptor"], "r": best_row["r"]})
        rows.append(row)

    results_df = pd.DataFrame(rows)

    written["metrics"] = out_dir / "metrics.csv"
    results_df.to_csv(written["metrics"], index=False)
    written["pc_correlations"] = out_dir / "pc_correlations.csv"
    pd.DataFrame(pc_corr_rows).to_csv(written["pc_correlations"], index=False)
    written["pc_best_descriptor"] = out_dir / "pc_best_descriptor.csv"
    pd.DataFrame(pc_best_rows).to_csv(written["pc_best_descriptor"], index=False)
    written["summary"] = out_dir / "summary.txt"
    written["summary"].write_text(_summary_report(results_df, property_names, include_silhouette))
    for key in ("metrics", "pc_correlations", "pc_best_descriptor", "summary"):
        print(f"Saved {written[key]}")

    # ---- PC-traversal filmstrips at a few selected layers ----
    decoder_names = [n for n in representations if n not in ("ECFP", "global_cond")]
    if traversal_layers:
        # A bare '6' means "layer 6 at its first available timestep". Unmatched
        # keys raise instead of being silently dropped.
        available_ts = sorted({ts for (_layer, ts) in aligned["layer_keys"]})
        layer_key_set = set(aligned["layer_keys"])
        selected_names, missing = [], []
        for raw, (layer, ts) in zip(traversal_layers, (_parse_traversal_key(s) for s in traversal_layers)):
            candidates = [ts] if ts is not None else available_ts
            match = next((c for c in candidates if (layer, c) in layer_key_set), None)
            if match is None:
                missing.append(raw)
            else:
                name = _slug_for(stream, layer, match)
                if name not in selected_names:
                    selected_names.append(name)
        if missing:
            raise SystemExit(
                f"--traversal-layers {missing} matched no extracted (layer, timestep) "
                f"combination for stream {stream!r}.\n"
                f"Available layers: {sorted({l for (l, _t) in aligned['layer_keys']})}\n"
                f"Available timesteps: {available_ts}\n"
                f"Pass entries as 'LAYER' or 'LAYER,TIMESTEP', e.g. "
                f"'6' or '6,{available_ts[0] if available_ts else 0.001}'."
            )
    elif decoder_names:
        sorted_decoder = sorted(decoder_names, key=lambda n: layer_of[n])
        selected_names = sorted(set([sorted_decoder[0], sorted_decoder[len(sorted_decoder) // 2],
                                     sorted_decoder[-1]]), key=lambda n: layer_of[n])
    else:
        selected_names = []
    print(f"\nPC-traversal filmstrips will be rendered for: {selected_names}")

    trav_rows: List[Dict[str, Any]] = []
    trav_stats_rows: List[Dict[str, Any]] = []
    for name in selected_names:
        x_scaled = StandardScaler().fit_transform(representations[name]).astype(np.float32)
        pca_fit = PCA(n_components=n_pcs_traversal, random_state=seed)
        pcs = pca_fit.fit_transform(x_scaled).astype(np.float32)

        for pc_idx in range(n_pcs_traversal):
            pc_values, step_indices, other_dim = traversal_steps(
                pcs, pc_idx, n_steps, pct_lo=pc_pct_lo, pct_hi=pc_pct_hi)
            for step, (val, idx) in enumerate(zip(pc_values, step_indices)):
                trav_rows.append({
                    "representation": name, "pc": pc_idx + 1, "step": step, "pc_value": float(val),
                    "row": int(idx), "smiles": smiles[idx],
                    "x": float(pcs[idx, pc_idx]), "y": float(pcs[idx, other_dim]),
                })
            trav_stats_rows.append({
                "representation": name, "pc": pc_idx + 1, "other_dim": other_dim + 1,
                "explained_variance_ratio": float(pca_fit.explained_variance_ratio_[pc_idx]),
            })

        backdrop = pcs
        if 0 < n_background_scatter < len(pcs):
            backdrop = pcs[rng.choice(len(pcs), n_background_scatter, replace=False)]
        bg_path = out_dir / f"traversal_background_{name}.npz"
        np.savez_compressed(bg_path, coords=backdrop)
        written[f"traversal_background_{name}"] = bg_path

    if trav_rows:
        written["traversal"] = out_dir / "traversal.csv"
        pd.DataFrame(trav_rows).to_csv(written["traversal"], index=False)
        written["traversal_stats"] = out_dir / "traversal_stats.csv"
        pd.DataFrame(trav_stats_rows).to_csv(written["traversal_stats"], index=False)
        print(f"Saved {written['traversal']}")
        print(f"Saved {written['traversal_stats']}")

    write_manifest(
        out_dir, slug=slug, tag=tag, schema_version=SCHEMA_VERSION,
        params={
            "prefix": prefix, "splits": list(splits), "data_dir": str(data_dir),
            "layerwise_dir": str(layerwise_dir), "stream": stream,
            "n_molecules": n_total,
            "representations": list(representations.keys()),
            "layer_keys": [[int(l), float(t)] for l, t in aligned["layer_keys"]],
            "property_names": property_names,
            "include_silhouette": include_silhouette,
            "knn_purity_k": knn_purity_k,
            "k_neighbors": list(k_neighbors),
            "nn_overlap_sample": nn_overlap_sample,
            "cluster_tag": cluster_tag,
            "have_cluster_labels": cluster_labels is not None,
            "descriptor_names": descriptor_names,
            "traversal_representations": selected_names,
            "n_pcs_traversal": n_pcs_traversal,
            "n_steps": n_steps,
            "pc_pct_lo": pc_pct_lo,
            "pc_pct_hi": pc_pct_hi,
            "n_background_scatter": n_background_scatter,
            "seed": seed,
            "device": device,
            "dataset_counts": {str(k): int(v) for k, v in
                               zip(*np.unique(aligned["dataset"], return_counts=True))},
        },
        inputs=[Path(layerwise_dir) / f"{prefix}_{s}_layerwise.pt" for s in splits]
              + [Path(data_dir) / f"{prefix}_{s}_global_cond.pt" for s in splits],
        outputs=list(written.values()),
    )
    return written
