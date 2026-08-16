"""One PCA over an embedding, explained in two vocabularies at once.

The previous pipeline fit this PCA twice -- once in `3_run_pca_experiment.py`
(explained with RDKit structural descriptors, filmstrips drawing molecule
STRUCTURES) and once in `17_run_spectral_pca_experiment.py` (explained with NMR
spectral features, filmstrips drawing SPECTRA). Same StandardScaler, same PCA,
same input. Their two "PC3" figures were only the same axis by coincidence of a
shared seed. Fitting once makes the structure filmstrip and the spectrum
filmstrip guaranteed views of one axis, showing one molecule per step.

What this module writes is *figure data*, not a research artifact dump. The old
`.pt` payload carried the raw N x 768 embedding, the full N x 35 descriptor
matrix, the full N x 37 spectral panel, and live pickled `PCA`/`StandardScaler`
objects -- gigabytes, unreadable without matching sklearn. Everything a figure
needs is decided here instead:

* traversal steps are resolved to real molecule indices, and only those
  molecules' SMILES, coordinates and peak lists are stored;
* the scatter backdrop is subsampled here, once;
* correlations are computed over every molecule, and only the columns some PC
  actually won are kept for the scatter panels.

Computation only -- this module never imports matplotlib.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.analysis.corpus import RaggedPeaks
from src.analysis.descriptors import (
    best_per_pc,
    compute_pc_correlations,
    correlation_report,
    top_per_pc,
)
from src.analysis.geometry import traversal_steps
from src.common.constants import DEFAULT_C_PPM_RANGE, DEFAULT_H_PPM_RANGE
from src.common.manifest import write_manifest


SCHEMA_VERSION = 1


def robust_normalize(features: np.ndarray, pct_lo: float = 1.0,
                     pct_hi: float = 99.0) -> np.ndarray:
    """Min-max each column onto [0, 1] using the [pct_lo, pct_hi] percentiles
    rather than true min/max.

    These are raw per-molecule spectral features with long tails -- one
    200-peak outlier against true min/max would squash every other bar to
    nothing -- so the range is trimmed and values clipped. Entirely-NaN columns
    come back NaN; constant ones come back 0.
    """
    with np.errstate(invalid="ignore"):
        finite = np.isfinite(features)
        safe = np.where(finite, features, np.nan)
        all_nan = ~finite.any(axis=0)
        lo = np.where(all_nan, 0.0, np.nanpercentile(np.where(all_nan, 0.0, safe), pct_lo, axis=0))
        hi = np.where(all_nan, 1.0, np.nanpercentile(np.where(all_nan, 1.0, safe), pct_hi, axis=0))
    span = hi - lo
    span = np.where(np.isfinite(span) & (span > 0), span, 1.0)
    return np.clip((safe - lo) / span, 0.0, 1.0)


def _best_columns(best_df: pd.DataFrame, name_col: str, all_names: List[str],
                  matrix: np.ndarray, n_pcs: int) -> np.ndarray:
    """(N, n_pcs) -- for each PC, the column of `matrix` it correlates best with.

    This is the whole reason the correlation figure does not need the full
    descriptor/spectral matrix in `data/`: each panel plots exactly one column,
    so only the winners have to survive.
    """
    out = np.full((matrix.shape[0], n_pcs), np.nan, dtype=np.float32)
    for i in range(min(n_pcs, len(best_df))):
        name = best_df.iloc[i][name_col]
        if name is None or (isinstance(name, float) and np.isnan(name)):
            continue
        out[:, i] = matrix[:, all_names.index(name)].astype(np.float32)
    return out


def fit_joint_pca(
    *,
    embedding: np.ndarray,
    smiles: Sequence[str],
    dataset: np.ndarray,
    descriptor_matrix: np.ndarray,
    descriptor_names: List[str],
    spectral_features: np.ndarray,
    spectral_feature_names: List[str],
    h_peaks: Optional[RaggedPeaks],
    c_peaks: Optional[RaggedPeaks],
    h_nh: Optional[RaggedPeaks],
    peak_index: Optional[np.ndarray],
    out_dir: Path,
    tag: str,
    slug: str,
    embedding_label: str,
    n_components: int = 8,
    scale: bool = True,
    seed: int = 1234,
    top_k: int = 5,
    n_steps: int = 8,
    pct_lo: float = 1.0,
    pct_hi: float = 99.0,
    n_background_scatter: int = 50_000,
    n_bar_features: int = 0,
    corr_scatter_max: int = 0,
    params: Optional[Dict[str, Any]] = None,
    inputs: Sequence[Path] = (),
) -> Dict[str, Path]:
    """Fits the PCA and writes everything the plotting script needs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}

    # A PCA over a handful of molecules returns components with NaN explained
    # variance and correlations computed from nothing -- output that looks like
    # a result. Refuse rather than emit it.
    if embedding.shape[0] < max(2 * n_components, 20):
        raise SystemExit(
            f"Only {embedding.shape[0]} molecules for {embedding_label} -- too few for a "
            f"{n_components}-component PCA. Check the split, the join, and any --limit used "
            f"at extraction time.")

    n_components = int(min(n_components, embedding.shape[0], embedding.shape[1]))
    x = embedding
    if scale:
        x = StandardScaler().fit_transform(embedding)
    pca = PCA(n_components=n_components, random_state=seed)
    pcs = pca.fit_transform(x).astype(np.float32)
    print(f"PCA: {n_components} components, explained variance ratio "
          f"{np.round(pca.explained_variance_ratio_, 4).tolist()}")

    # --- explanation 1: RDKit structural descriptors -------------------------
    desc_corr = compute_pc_correlations(pcs, descriptor_matrix, descriptor_names,
                                        desc="PC-descriptor correlations")
    best_desc = best_per_pc(desc_corr, "descriptor")

    # --- explanation 2: NMR spectral features --------------------------------
    spec_corr = compute_pc_correlations(pcs, spectral_features, spectral_feature_names,
                                        desc="PC-spectral correlations")
    best_spec = best_per_pc(spec_corr, "feature")
    top_spec = top_per_pc(spec_corr, top_k=top_k, column="feature")

    # The two frames keep DISTINCT names on disk. Both source scripts called
    # their own frame "correlations", so a merged artifact reusing that name
    # would silently hand the spectral heatmap the structural matrix.
    written["descriptor_correlations"] = out_dir / "descriptor_correlations.csv"
    desc_corr.to_csv(written["descriptor_correlations"])
    (out_dir / "descriptor_correlations.txt").write_text(
        correlation_report(desc_corr, "RDKit descriptor"))
    written["pc_best_descriptor"] = out_dir / "pc_best_descriptor.csv"
    best_desc.to_csv(written["pc_best_descriptor"], index=False)

    written["spectral_correlations"] = out_dir / "spectral_correlations.csv"
    spec_corr.to_csv(written["spectral_correlations"])
    (out_dir / "spectral_correlations.txt").write_text(
        correlation_report(spec_corr, "NMR spectral feature"))
    written["pc_best_spectral_feature"] = out_dir / "pc_best_spectral_feature.csv"
    best_spec.to_csv(written["pc_best_spectral_feature"], index=False)
    written["pc_top_spectral_features"] = out_dir / "pc_top_spectral_features.csv"
    top_spec.to_csv(written["pc_top_spectral_features"], index=False)

    # --- correlation-scatter data -------------------------------------------
    # One column per PC on each side: the descriptor/feature that PC actually
    # won. `corr_scatter_max` thins the DRAWN points only -- every r above was
    # computed over all molecules.
    rng = np.random.default_rng(seed)
    n_total = len(smiles)
    if 0 < corr_scatter_max < n_total:
        keep = np.sort(rng.choice(n_total, corr_scatter_max, replace=False))
        print(f"Correlation scatters thinned to {corr_scatter_max}/{n_total} drawn points "
              f"(reported r values still use all {n_total}).")
    else:
        keep = np.arange(n_total)

    np.savez_compressed(
        out_dir / "corr_scatter.npz",
        pc_values=pcs[keep],
        descriptor_values=_best_columns(best_desc, "descriptor", descriptor_names,
                                        descriptor_matrix, n_components)[keep],
        spectral_values=_best_columns(best_spec, "feature", spectral_feature_names,
                                      spectral_features, n_components)[keep],
        n_total=np.asarray(n_total),
    )
    written["corr_scatter"] = out_dir / "corr_scatter.npz"

    # --- traversal ----------------------------------------------------------
    have_peaks = h_peaks is not None and c_peaks is not None
    normalized = None
    if n_bar_features > 0:
        normalized = robust_normalize(spectral_features, pct_lo, pct_hi)

    stats_rows: List[Dict[str, Any]] = []
    trav_rows: List[Dict[str, Any]] = []
    bar_rows: List[Dict[str, Any]] = []
    step_corpus_rows: List[int] = []

    for pc_idx in range(n_components):
        pc_values, step_indices, other_dim = traversal_steps(
            pcs, pc_idx, n_steps, pct_lo=pct_lo, pct_hi=pct_hi)

        bar_features: List[str] = []
        if normalized is not None:
            row = spec_corr.iloc[pc_idx]
            ranked = row.abs().sort_values(ascending=False)
            bar_features = [f for f in ranked.index if np.isfinite(row[f])][:n_bar_features]

        h_y_max = 1.0
        for step, (val, idx) in enumerate(zip(pc_values, step_indices)):
            corpus_row = int(peak_index[idx]) if peak_index is not None else int(idx)
            n_h = n_c = 0
            # `peak_row` indexes traversal_peaks.npz, which stores only the step
            # molecules in the order they are appended here -- so the plotting
            # side never needs the corpus-wide ragged store.
            peak_row = -1
            if have_peaks:
                nh_arr = h_nh[corpus_row] if h_nh is not None else None
                h_arr = h_peaks[corpus_row]
                n_h, n_c = len(h_arr), len(c_peaks[corpus_row])
                if nh_arr is not None and len(nh_arr):
                    finite = nh_arr[np.isfinite(nh_arr) & (nh_arr > 0)]
                    if len(finite):
                        h_y_max = max(h_y_max, float(finite.max()))
                peak_row = len(step_corpus_rows)
                step_corpus_rows.append(corpus_row)

            trav_rows.append({
                "pc": pc_idx + 1, "step": step, "pc_value": float(val),
                "row": int(idx), "peak_row": peak_row, "smiles": smiles[idx],
                "x": float(pcs[idx, pc_idx]), "y": float(pcs[idx, other_dim]),
                "n_h_peaks": n_h, "n_c_peaks": n_c,
            })
            for feature in bar_features:
                bar_rows.append({
                    "pc": pc_idx + 1, "step": step, "feature": feature,
                    "value": float(normalized[idx, spectral_feature_names.index(feature)]),
                })

        stats_rows.append({
            "pc": pc_idx + 1,
            "other_dim": other_dim + 1,
            "explained_variance_ratio": float(pca.explained_variance_ratio_[pc_idx]),
            "pct_lo_value": float(pc_values[0]),
            "pct_hi_value": float(pc_values[-1]),
            "median": float(np.median(pcs[:, pc_idx])),
            # Filmstrip-wide 1H scale, so stick heights mean the same thing in
            # every panel instead of each renormalizing itself.
            "h_y_max": h_y_max,
        })

    written["pc_stats"] = out_dir / "pc_stats.csv"
    pd.DataFrame(stats_rows).to_csv(written["pc_stats"], index=False)
    written["traversal"] = out_dir / "traversal.csv"
    pd.DataFrame(trav_rows).to_csv(written["traversal"], index=False)
    if bar_rows:
        written["traversal_bars"] = out_dir / "traversal_bars.csv"
        pd.DataFrame(bar_rows).to_csv(written["traversal_bars"], index=False)

    if have_peaks and step_corpus_rows:
        h_vals, h_off = h_peaks.take(step_corpus_rows)
        c_vals, c_off = c_peaks.take(step_corpus_rows)
        arrays = {"h_values": h_vals, "h_offsets": h_off,
                  "c_values": c_vals, "c_offsets": c_off}
        if h_nh is not None:
            nh_vals, _ = h_nh.take(step_corpus_rows)
            arrays["h_nh_values"] = nh_vals
        np.savez_compressed(out_dir / "traversal_peaks.npz", **arrays)
        written["traversal_peaks"] = out_dir / "traversal_peaks.npz"
    else:
        print("[note] No raw peak lists available -- the joint traversal filmstrips will be "
              "skipped. Re-run extraction/01_spectral_features.py so its output carries "
              "h_peak_values/h_peak_offsets/c_peak_values/c_peak_offsets. Correlation "
              "figures are unaffected.")

    # --- scatter backdrop ----------------------------------------------------
    # The same cloud is repeated in every panel of every filmstrip, so drawing a
    # multi-million-point corpus into all of them is pure cost -- at 0.3pt /
    # alpha 0.2 it saturates long before that. Percentiles, medians and the
    # nearest-molecule search above all used every point.
    backdrop = pcs
    if 0 < n_background_scatter < len(pcs):
        backdrop = pcs[rng.choice(len(pcs), n_background_scatter, replace=False)]
    np.savez_compressed(out_dir / "pcs_background.npz", coords=backdrop.astype(np.float32))
    written["pcs_background"] = out_dir / "pcs_background.npz"

    write_manifest(
        out_dir, slug=slug, tag=tag, schema_version=SCHEMA_VERSION,
        params={
            **(params or {}),
            "embedding": embedding_label,
            "n_molecules": int(n_total),
            "n_components": n_components,
            "scale": bool(scale),
            "seed": seed,
            "n_steps": n_steps,
            "pct_lo": pct_lo,
            "pct_hi": pct_hi,
            "n_bar_features": n_bar_features,
            "n_background_scatter": n_background_scatter,
            "corr_scatter_max": corr_scatter_max,
            "corr_scatter_drawn": int(len(keep)),
            "n_descriptors": len(descriptor_names),
            "n_spectral_features": len(spectral_feature_names),
            "have_peaks": bool(have_peaks),
            "embedding_dim": int(embedding.shape[1]),
            "h_ppm_range": list(DEFAULT_H_PPM_RANGE),
            "c_ppm_range": list(DEFAULT_C_PPM_RANGE),
            "dataset_counts": {str(k): int(v) for k, v in
                               zip(*np.unique(dataset, return_counts=True))},
        },
        inputs=list(inputs),
        outputs=list(written.values()),
    )
    return written
