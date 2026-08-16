#!/usr/bin/env python
"""
umap_shared_cache.py
=====================

Shared UMAP fitting + hyperparameter-keyed disk cache used by every script
in this pipeline that plots a 2D UMAP projection of the cotrain embeddings
(5_run_pretty_plots_experiment.py / 6_run_pretty_plots_batch_experiment.py /
7_run_umap_experiment.py / 8_run_real_vs_synthetic_experiment.py, and any
future script that needs the same 2D space).

Why this exists
----------------
The pretty-plot and property-plot scripts each used to carry their own UMAP
cache: one keyed by a hyperparameter fingerprint (-> cached .npy coordinates
+ pickled reducer), the other a completely separate, incompatible scheme
(--save-umap-model/--load-umap-model, no fingerprinting, and critically no
scale/PCA preprocessing). That meant a fit made by one could never be reused
by the other even when every hyperparameter "matched", and the *shape* of the
two UMAPs could differ for the nominally-same embedding, because only one of
them applied the StandardScaler + PCA preprocessing before fitting.

This module is now the single source of truth for that pipeline. Every
script fits UMAP the same way (StandardScaler -> PCA -> UMAP) and reads/
writes the same on-disk cache format, keyed by the same hyperparameter
fingerprint. Point multiple scripts at the same `--umap-cache-dir` (default:
`./umap_shared_cache`, shared by all of them unless overridden -- see
DEFAULT_CACHE_DIR below) and a UMAP fit made by one script -- e.g. script
6's default global_cond run -- is reused verbatim by any other script that
asks for the same hyperparameters, INCLUDING scripts (like 7) that need to
`.transform()` brand-new points (val/test) into that exact same 2D space,
not just reload cached coordinates for the original point set.

Fitting pipeline
-----------------
    embedding --[StandardScaler, optional]--> --[PCA, optional]--> --[UMAP]--> 2D

Cache format (all under a single `--umap-cache-dir`)
------------------------------------------------------
`<cache-dir>/manifest.json`: a list of records, one per distinct
hyperparameter fingerprint:
    {"fingerprint", "coords_path", "bundle_path", "created_at", **fields}
`fields` is the exact dict that was fingerprinted (embedding_key, datasets,
n_neighbors, min_dist, metric, pca_dim, scale, seed, n_points, n_dims) --
kept alongside the hash purely so the manifest is human-readable/greppable;
matching is always done on `fingerprint`, never by re-comparing `fields`.

`<cache-dir>/umap_<fp>.npy`: the fitted 2D coordinates for the exact point
set that was fit (same row order as the input embedding).

`<cache-dir>/umap_<fp>_bundle.pkl`: a pickled `UmapFitBundle` -- the fitted
StandardScaler, PCA, and UMAP reducer objects (whichever of the first two
were actually used, per `scale`/`pca_dim`). This is what lets a DIFFERENT
set of points (e.g. val/test, which weren't part of the original fit) get
mapped into the exact same 2D space later, via `transform_new_points()`,
rather than only ever being able to replot the original training points.
Bundle pickling can occasionally fail (e.g. incompatible library versions
between machines); if so, the cache still stores coordinates for the
original points, but `transform_new_points` won't be usable for that
fingerprint until it's refit (this is handled automatically by
`resolve_umap(..., need_bundle=True)`; see its docstring).

NOTE on fingerprint semantics (unchanged from before): the fingerprint is a
CONFIG fingerprint (hyperparameters + point count + dimensionality), NOT a
content hash of the embedding matrix itself. Swapping in different data with
the same shape/config will false-positive match a stale cache. Use a
separate --umap-cache-dir per dataset/experiment if that's a concern.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------------
# Shared defaults. Imported by scripts 5/6/7/8 so their UMAP hyperparameter
# defaults (and therefore their cache fingerprints) can never silently drift
# apart from each other. Any future script that fits or reuses this UMAP
# should import these too rather than redefining them.
# -----------------------------------------------------------------------------

DEFAULT_CACHE_DIR = Path("umap_shared_cache")
DEFAULT_UMAP_N_NEIGHBORS = 30
DEFAULT_UMAP_MIN_DIST = 0.1
DEFAULT_PCA_DIM = 50
DEFAULT_UMAP_SEED = 0

EMBEDDING_KEYS = ["ecfp", "global_cond", "decoder"]

DEFAULT_METRIC_BY_EMBEDDING = {
    "ecfp": "jaccard",
    "global_cond": "cosine",
    "decoder": "cosine",
}

# StandardScaler doesn't make sense for a binary fingerprint used with a
# fingerprint-oriented metric; it's fine (and the pipeline-wide default) for
# continuous embeddings.
DEFAULT_SCALE_BY_EMBEDDING = {
    "ecfp": False,
    "global_cond": True,
    "decoder": True,
}

# Metrics for which scaling a binary vector doesn't make sense (used for the
# scale/metric mismatch warning in maybe_scale()).
BINARY_FINGERPRINT_METRICS = {"jaccard", "hamming", "dice", "matching", "russellrao"}


@dataclass
class UmapFitBundle:
    """Everything needed to map NEW points into an existing UMAP fit's 2D
    space via transform_new_points(): the fitted preprocessing steps plus
    the fitted reducer itself. `scaler`/`pca` are None if that step was
    skipped (scale=False / pca_dim=0) at fit time -- transform_new_points
    skips them identically, so it never silently applies a preprocessing
    step the fit itself didn't use."""
    scaler: Optional[StandardScaler]
    pca: Optional[PCA]
    reducer: object  # umap.UMAP; typed as `object` so importing this module
                      # doesn't require `umap` to be installed just to read
                      # a cached bundle's metadata.
    fields: Dict


# -----------------------------------------------------------------------------
# Fingerprinting + manifest
# -----------------------------------------------------------------------------

def umap_fingerprint(fields: Dict) -> str:
    canonical = json.dumps(fields, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _load_manifest(cache_dir: Path) -> List[Dict]:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        return []
    return json.loads(manifest_path.read_text())


def _save_manifest(cache_dir: Path, manifest: List[Dict]) -> None:
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def _resolve_cached_path(cache_dir: Path, value: Optional[str]) -> Optional[Path]:
    """Manifest paths may be relative to the repo that wrote them.

    The previous pipeline recorded e.g. `umap_shared_cache/umap_<fp>.npy`,
    relative to ITS repo root. Read from anywhere else those resolve against the
    wrong directory and every cache hit looks like a miss -- which silently
    turns a free lookup into a multi-hour refit. Relative entries are therefore
    resolved against the cache directory's parent, which is exactly the root
    they were written relative to.
    """
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else (cache_dir.parent / path)


def find_cached_record(cache_dir: Path, fingerprint: str) -> Optional[Dict]:
    """Looks up one fingerprint, returning a record whose paths are absolute."""
    for record in _load_manifest(cache_dir):
        if record.get("fingerprint") == fingerprint:
            coords_path = _resolve_cached_path(cache_dir, record.get("coords_path"))
            if coords_path is not None and coords_path.exists():
                resolved = dict(record)
                resolved["coords_path"] = str(coords_path)
                bundle_path = _resolve_cached_path(cache_dir, record.get("bundle_path"))
                resolved["bundle_path"] = str(bundle_path) if bundle_path else None
                return resolved
            print(f"[umap cache] Manifest points at cached coordinates that "
                  f"no longer exist on disk ({coords_path}) -- refitting.")
    return None


def find_cached_record_in(cache_dirs: Sequence[Optional[Path]],
                          fingerprint: str) -> Tuple[Optional[Dict], Optional[Path]]:
    """First matching record across several caches, with the dir it came from.

    Lets a run write into its own cache while still reading fits computed by an
    earlier pipeline, without copying gigabytes of bundles around.
    """
    for cache_dir in cache_dirs:
        if cache_dir is None or not Path(cache_dir).exists():
            continue
        record = find_cached_record(Path(cache_dir), fingerprint)
        if record is not None:
            return record, Path(cache_dir)
    return None, None


def register_cache(cache_dir: Path, fingerprint: str, coords_path: Path, fields: Dict,
                    bundle_path: Optional[Path]) -> None:
    manifest = _load_manifest(cache_dir)
    manifest = [r for r in manifest if r.get("fingerprint") != fingerprint]
    manifest.append({
        "fingerprint": fingerprint, "coords_path": str(coords_path),
        "bundle_path": str(bundle_path) if bundle_path is not None else None,
        "created_at": datetime.now(timezone.utc).isoformat(), **fields,
    })
    _save_manifest(cache_dir, manifest)


# -----------------------------------------------------------------------------
# Fitting pipeline: scale -> PCA -> UMAP
# -----------------------------------------------------------------------------

def maybe_scale(embedding: np.ndarray, scale: bool, metric: str
                 ) -> Tuple[np.ndarray, Optional[StandardScaler]]:
    if not scale:
        return embedding, None
    if metric.lower() in BINARY_FINGERPRINT_METRICS:
        print(f"[warn] scale is on but metric='{metric}' is a binary-"
              f"fingerprint-oriented metric -- StandardScaler will turn 0/1 "
              f"bits into continuous z-scores, which usually isn't what you "
              f"want for that metric. Consider disabling scaling.")
    print(f"Applying StandardScaler to the embedding ({embedding.shape[0]} "
          f"points, {embedding.shape[1]} dims) before PCA/UMAP ...")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(embedding).astype(np.float32)
    return scaled, scaler


def reduce_with_pca(embedding: np.ndarray, n_components: Optional[int], seed: int
                     ) -> Tuple[np.ndarray, Optional[PCA]]:
    """Reduces to n_components via PCA before UMAP. Skipped if n_components
    is None/0, or if the embedding already has <= n_components dimensions
    (PCA can't add dimensions, and there's nothing to gain)."""
    if not n_components:
        return embedding, None
    if embedding.shape[1] <= n_components:
        print(f"Skipping PCA: embedding already has {embedding.shape[1]} dims "
              f"(<= requested {n_components}).")
        return embedding, None

    print(f"Reducing {embedding.shape[1]} dims -> {n_components} dims via PCA "
          f"before UMAP (on {embedding.shape[0]} points) ...")
    pca = PCA(n_components=n_components, random_state=seed)
    reduced = pca.fit_transform(embedding).astype(np.float32)
    explained = pca.explained_variance_ratio_.sum()
    print(f"PCA done. {n_components} components explain "
          f"{explained * 100:.1f}% of variance.")
    return reduced, pca


def fit_umap(embedding: np.ndarray, n_neighbors: int, min_dist: float, metric: str,
             seed: int, pca_dim: Optional[int], scale: bool, fields: Dict
             ) -> Tuple[np.ndarray, "UmapFitBundle"]:
    """Fits scale -> PCA -> UMAP fresh (no cache lookup -- see resolve_umap()
    for the cached entry point every script should actually call) and
    returns both the 2D coordinates and a bundle that can later map NEW
    points into this exact same space."""
    import umap.umap_ as umap_lib

    scaled, scaler = maybe_scale(embedding, scale, metric)
    reduced, pca = reduce_with_pca(scaled, pca_dim, seed)

    if pca is not None and metric not in ("euclidean", "cosine"):
        print(f"[warn] metric='{metric}' was requested for UMAP, but the "
              f"input to UMAP is now PCA-reduced (dense, continuous) rather "
              f"than the original representation. Binary-fingerprint-"
              f"oriented metrics like 'jaccard' don't really apply to PCA "
              f"output -- consider metric='cosine' or 'euclidean' instead, "
              f"or pca_dim=0 to disable PCA and keep the original metric "
              f"meaningful.")

    print(f"Fitting UMAP (n_neighbors={n_neighbors}, min_dist={min_dist}, "
          f"metric={metric}) on {reduced.shape[0]} points ...")
    reducer = umap_lib.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                             metric=metric, n_jobs=-1, random_state=seed)
    emb2d = reducer.fit_transform(reduced)

    bundle = UmapFitBundle(scaler=scaler, pca=pca, reducer=reducer, fields=fields)
    return emb2d, bundle


def transform_new_points(bundle: UmapFitBundle, embedding: np.ndarray) -> np.ndarray:
    """Maps NEW points (e.g. val/test -- not part of the original fit) into
    the 2D space of an existing UmapFitBundle, applying the exact same
    scale -> PCA -> UMAP.transform pipeline used at fit time (skipping
    whichever steps were skipped then, so this can never apply a
    preprocessing step the original fit didn't use)."""
    x = embedding
    if bundle.scaler is not None:
        x = bundle.scaler.transform(x).astype(np.float32)
    if bundle.pca is not None:
        x = bundle.pca.transform(x).astype(np.float32)
    return bundle.reducer.transform(x)


def save_bundle(path: Path, bundle: UmapFitBundle) -> Optional[Path]:
    """Best-effort pickle of the fit bundle. Returns None (with a warning,
    not an exception) if pickling fails, so callers can still keep the
    cached coordinates even when the bundle can't be saved."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(bundle, f)
        return path
    except Exception as e:
        print(f"[warn] Failed to pickle the UMAP fit bundle ({e}) -- caching "
              f"coordinates only; other scripts won't be able to "
              f".transform() new points into this fit. It'll be refit (and "
              f"a fresh bundle attempted again) the next time something "
              f"needs one.")
        return None


def load_bundle(path: Path) -> UmapFitBundle:
    with open(path, "rb") as f:
        return pickle.load(f)


# -----------------------------------------------------------------------------
# Main entry point: cached resolve
# -----------------------------------------------------------------------------

def resolve_umap(embedding: np.ndarray, fingerprint_fields: Dict, *, n_neighbors: int,
                  min_dist: float, metric: str, seed: int, pca_dim: Optional[int],
                  scale: bool, cache_dir: Optional[Path],
                  load_coords_path: Optional[Path] = None,
                  save_coords_path: Optional[Path] = None,
                  need_bundle: bool = False,
                  read_only_cache_dirs: Sequence[Optional[Path]] = (),
                  ) -> Tuple[np.ndarray, Optional[UmapFitBundle]]:
    """The single entry point every script in this pipeline should use to
    get a 2D UMAP projection. Returns (emb2d, bundle_or_None).

    - `load_coords_path`, if given, takes precedence over everything else
      (an explicit precomputed-projection override). Note it never provides
      a bundle: a `need_bundle=True` caller gets `(coords, None)` back and
      must handle that itself (e.g. skip transforming new points, or don't
      pass an explicit path if you need to transform() new points -- go
      through `cache_dir` instead).
    - Otherwise: fingerprints `fingerprint_fields`, checks `cache_dir` for a
      match.
        - Hit: loads the cached coordinates. If `need_bundle=True`, ALSO
          tries to load the cached bundle -- if that specific cache entry
          doesn't have a usable bundle (e.g. bundle-pickling failed when it
          was written), this refits fresh instead of silently returning
          `(coords, None)`, so a `need_bundle=True` caller always either
          gets a working bundle or an explicit fresh fit.
        - Miss (or the forced refit above): fits fresh via `fit_umap`, and
          always saves both coordinates AND bundle back into `cache_dir`
          (when given) for next time -- regardless of whether THIS caller
          needed the bundle, since a later caller (e.g.
          script 7, run after script 6) might.
    """
    if load_coords_path is not None:
        print(f"Loading precomputed 2D projection from {load_coords_path}")
        emb2d = np.load(load_coords_path)
        if emb2d.shape[0] != embedding.shape[0]:
            raise ValueError(
                f"{load_coords_path} has {emb2d.shape[0]} rows but the "
                f"input embedding has {embedding.shape[0]} rows -- a "
                f"precomputed projection must be aligned to the exact "
                f"point set being plotted."
            )
        if need_bundle:
            print("[warn] An explicit --umap-embedding-path was given, so "
                  "no fit bundle is available -- new points (e.g. val/"
                  "test) can't be .transform()'d into this projection.")
        return emb2d, None

    fp = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        fp = umap_fingerprint(fingerprint_fields)
        record, found_in = find_cached_record_in([cache_dir, *read_only_cache_dirs], fp)
        if record is not None and found_in != cache_dir:
            print(f"[umap cache] Reusing a fit from the read-only cache {found_in}.")
        if record is not None:
            bundle = None
            if need_bundle:
                bundle_path = record.get("bundle_path")
                if bundle_path is not None and Path(bundle_path).exists():
                    bundle = load_bundle(Path(bundle_path))
                else:
                    print(f"[umap cache] Cached fit (fingerprint={fp}) has no "
                          f"usable bundle, but a bundle was requested (need "
                          f"to .transform() new points) -- refitting so a "
                          f"bundle gets cached this time.")
                    record = None
            if record is not None:
                emb2d = np.load(record["coords_path"])
                print(f"Found cached UMAP run matching these hyperparameters "
                      f"(fingerprint={fp}): {record['coords_path']} -- "
                      f"loading instead of refitting.")
                return emb2d, bundle
        else:
            print(f"No cached UMAP run found for fingerprint={fp}; fitting fresh.")

    emb2d, bundle = fit_umap(embedding, n_neighbors, min_dist, metric, seed,
                              pca_dim, scale, fingerprint_fields)

    if cache_dir is not None:
        # Absolute, so a manifest written here stays readable from any cwd or
        # from another repo pointing at this cache read-only.
        cache_dir = Path(cache_dir).resolve()
        coords_path = cache_dir / f"umap_{fp}.npy"
        np.save(coords_path, emb2d)
        bundle_path = save_bundle(cache_dir / f"umap_{fp}_bundle.pkl", bundle)
        register_cache(cache_dir, fp, coords_path, fingerprint_fields, bundle_path)
        print(f"✓ Cached this UMAP run for future reuse: {coords_path}"
              + (f" (+ fit bundle: {bundle_path})" if bundle_path else ""))

    if save_coords_path is not None:
        save_coords_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_coords_path, emb2d)
        print(f"✓ Also saved to {save_coords_path}")

    return emb2d, bundle
