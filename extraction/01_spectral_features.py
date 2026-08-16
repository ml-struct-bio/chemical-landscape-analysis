#!/usr/bin/env python
"""
01_spectral_features.py
============================

Precomputes the per-molecule SPECTRAL feature panel for the whole cotrain
corpus, aligned one-to-one with the embeddings `extraction/00_global_cond.py`
already wrote, and saves it beside them as

    <prefix>_<split>_spectral_features.pt

Why this exists as a separate script
------------------------------------
Spectral features are derived from the raw NMR peak lists. Those peaks are
visible only while a datamodule batch is in hand -- `00_global_cond.py` sees
them, uses them for the forward pass, and discards them, storing embeddings/
SMILES/dataset/ECFP. Its `condition` entry is the condition *name* ("hcpeak"),
not the peak dict. So anything downstream that wants to colour, correlate, or
filter by spectral features had to re-derive them behind a checkpoint, which is
what forced the spectral-PCA analysis onto a GPU in the previous pipeline.

This script closes that gap without touching `00_global_cond.py`, which is
deliberately kept matching the original pipeline. It saves both the feature
panel AND the raw peak lists, which is what makes a spectral-space PCA
structurally identical to a structure-space one: the structure traversals draw
molecule STRUCTURES, and RDKit renders those from the SMILES step 0 already
caches; the spectral traversals draw SPECTRA, which need peaks. With peaks
cached here, that analysis can traverse and film the full corpus on CPU.

**No GPU, no checkpoint forward pass.** The peaks come straight off the
datamodule; the model is never constructed. The checkpoint is only *read* (on
CPU) for its preprocessing settings, and in fact not even that -- those settings
are recovered from step `0`'s own manifest, see below.

How alignment is guaranteed
---------------------------
Reproducing step `0`'s molecule ORDER is necessary but not sufficient, so this
script does not assume it:

  * Configuration is read from `<prefix>_manifest.json` rather than restated
    here -- the same sources, hydra names, split suffixes, valid_h_key/
    valid_c_key, `c_peak_norm_args`, and padding bounds that produced the
    embeddings. A source list duplicated in two scripts drifts; a manifest
    cannot.
  * Molecules are then matched to step `0`'s saved `smiles`/`dataset` lists.
    A positional check runs first (train/val are not subsampled, so the
    dataloader order should already agree); if it holds, the panel is used
    as-is. Otherwise rows are looked up by `(dataset, SMILES)`, which is
    unambiguous on this corpus -- all 116,043 val molecules have distinct
    (dataset, SMILES) pairs.
  * The output is written in step `0`'s order, one row per embedding, with
    NaN rows for anything that could not be matched, and the match rate is
    reported and stored in the artifact. Row i of `spectral_features` always
    corresponds to row i of `global_cond`.

The `test` split needs the cap lifted
-------------------------------------
`nmr-to-3d/configs/config.yaml` sets `dataset_args.test_args.test_samples: 100`
with a null `test_seed`, and `NMRDataModule.setup("test")` applies it
unconditionally. Step `0` does not override it, so its test split is a RANDOM
100 molecules per source (300 total) drawn with `default_rng(None)` -- a
different 100 on every run. Re-extracting the test split therefore cannot
reproduce step `0`'s sample positionally.

This script lifts the cap (`test_samples` set past the split size) so it
extracts the FULL test split, which is a superset of whatever 100 step `0`
happened to draw, and recovers step `0`'s exact molecules from it by the
`(dataset, SMILES)` lookup above. Train/val are not capped and are unaffected.

Output
------
    <save-dir>/<prefix>_<split>_spectral_features.pt
    {
        "split": <str>,
        "prefix": <str>,
        "spectral_features": FloatTensor [N, 37],   # N == len(step 0's global_cond)
        "spectral_feature_names": List[str],
        # Ragged peak lists -- molecule i's are values[offsets[i]:offsets[i+1]].
        # Stored flat rather than padded (0.42 GB vs 2.4 GB on this corpus) and
        # rather than as millions of small arrays. This is what lets the spectral-PCA analysis
        # draw its traversal filmstrips with no checkpoint.
        "h_peak_values": FloatTensor [sum(n_H)],      # 1H shifts, ppm
        "h_peak_nh_values": FloatTensor [sum(n_H)],   # per-peak integration
        "h_peak_offsets": LongTensor [N+1],
        "c_peak_values": FloatTensor [sum(n_C)],      # 13C shifts, ppm
        "c_peak_offsets": LongTensor [N+1],
        "smiles": List[str],        # copied from step 0, for re-verification
        "dataset": List[str],       # ditto
        "n_matched": int,           # rows that got real values
        "n_missing": int,           # rows left as NaN
        "alignment": "positional" | "smiles_join",
        "h_reference": "total",
        "corpus_file": <path to the step-0 file this is aligned to>,
        "corpus_manifest": {...},   # the manifest stanza it was configured from
        "extracted_at": ISO timestamp,
    }

Usage
-----
    python extraction/01_spectral_features.py \
        --nmr3d-root /abs/path/to/nmr-to-3d \
        --corpus-dir /scratch/.../26-07-27-cotrain-v3-dedupOFF-s0/epoch1399 \
        --prefix cotrain --splits train val test

Memory: reading step `0`'s output to get its SMILES pulls the whole file in,
including the embedding tensor (~13 GB for this corpus's train split). Budget
accordingly, or run one split per job with `--splits`.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

# This script lives in extraction/, so sys.path[0] is that directory rather than
# the repo root and a bare `from src....` would not resolve.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# These came from src/analysis/real_vs_synthetic_analysis.py in the previous
# pipeline. The code is unchanged, but it now lives under src/extraction/ --
# both because that is what it is, and because importing it from the old module
# pulled in matplotlib, umap_shared_cache and the palette as a side effect, none
# of which this CPU-only script has any use for.
from src.extraction.nmr3d import build_datamodule, get_split_dataloader
from src.extraction.spectral_features import (
    QUALITY_FEATURE_NAMES,
    STRUCTURE_FEATURE_NAMES,
    DEFAULT_H_REFERENCE,
    _quality_features,
    _structure_reference,
    compute_spectral_feature_matrix,
)
from src.common.workers import safe_n_workers

# Condition keys the panel is built from -- the `hcpeak` condition's own names
# (nmr-to-3d/configs/condition/hcpeak.yaml), matching the spectral-PCA analysis's defaults.
H_SHIFT_KEY = "h_peak_centroid"
C_SHIFT_KEY = "c_peak_centroid"
H_MASK_KEY = "h_peak_mask"
C_MASK_KEY = "c_peak_mask"
H_NH_KEY = "h_peak_nH"


def parse_args():
    p = argparse.ArgumentParser(
        description="Precompute the spectral feature panel for the cotrain corpus, aligned "
                    "1:1 with extraction/00_global_cond.py's embeddings. CPU-only.")
    p.add_argument("--nmr3d-root", type=Path, required=True,
                   help="Path to the nmr-to-3d checkout (hydra configs + datasets).")
    p.add_argument("--corpus-dir", type=Path, required=True,
                   help="Directory holding step 0's <prefix>_<split>_global_cond.pt and "
                        "<prefix>_manifest.json. Read for configuration AND for the "
                        "molecule order to align to.")
    p.add_argument("--prefix", type=str, default="cotrain")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                   choices=["train", "val", "test"])
    p.add_argument("--save-dir", type=Path, default=None,
                   help="Where to write <prefix>_<split>_spectral_features.pt. "
                        "Default: --corpus-dir, i.e. beside the embeddings.")
    p.add_argument("--n-workers", type=int, default=safe_n_workers(),
                   help="Workers for the RDKit structure-reference pass.")
    p.add_argument("--limit", type=int, default=None,
                   help="Debug only: stop each source/split after ~this many molecules. "
                        "Produces a deliberately incomplete panel (the unmatched rows stay "
                        "NaN), so don't use it for a real run.")
    p.add_argument("--h-reference", type=str, default=DEFAULT_H_REFERENCE,
                   choices=["total", "carbon_bound"],
                   help="Which protons H_proton_balance is measured against. 'total' is "
                        "correct for this corpus (its peak lists include exchangeables).")
    return p.parse_args()


# -----------------------------------------------------------------------------
# Configuration recovered from step 0's manifest
# -----------------------------------------------------------------------------


def read_corpus_manifest(corpus_dir: Path, prefix: str) -> Dict[str, Any]:
    path = corpus_dir / f"{prefix}_manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Expected step 0's manifest at {path}. This script configures itself from it "
            "(sources, split suffixes, valid_h_key, c_peak_norm_args, padding bounds) so the "
            "peaks are preprocessed exactly as they were when the embeddings were made.")
    manifest = json.loads(path.read_text())
    for key in ("sources", "condition", "ckpt_sigma"):
        if key not in manifest:
            raise KeyError(f"{path} has no '{key}' -- not a step-0 manifest?")
    return manifest


def source_overrides(source: Dict[str, Any]) -> List[str]:
    """Hydra overrides for one source, exactly as step 0 applied them.

    `extra_overrides` in the manifest already carries the checkpoint-derived
    `c_peak_norm_args` and padding bounds. `valid_h_key`/`valid_c_key` are
    stored separately because step 0 passes them as its own arguments -- they
    matter: spectranp's default `valid_indices_h` lets 2 corrupt-1H molecules
    (nH=189/177) through, which is why the v3 config points at
    `valid_indices_h_clean` instead. Skipping them silently changes the
    molecule set and breaks alignment.
    """
    overrides = list(source.get("extra_overrides") or [])
    if source.get("valid_h_key"):
        overrides.append(f"dataset_args.valid_h_key={source['valid_h_key']}")
    if source.get("valid_c_key"):
        overrides.append(f"dataset_args.valid_c_key={source['valid_c_key']}")
    return overrides


def test_split_overrides(split: str, seed: int) -> List[str]:
    """Lift the 100-molecule test cap so the full test split is extracted.

    See the module docstring: step 0's test split is a random 100 per source and
    cannot be reproduced positionally, so this extracts the whole split and lets
    the SMILES join recover step 0's molecules from it.
    """
    if split != "test":
        return []
    return [f"dataset_args.test_args.test_samples={10**9}",
            f"dataset_args.test_args.test_seed={seed}"]


# -----------------------------------------------------------------------------
# Peak extraction (no model, no GPU)
# -----------------------------------------------------------------------------


def extract_source_split(dm, split: str, limit: Optional[int]) -> Optional[Dict[str, Any]]:
    """Iterates the dataloader for peaks only and reduces each batch to features.

    Deliberately never touches the model: `model_inputs["condition"]` is the
    peak dict, which is all the panel needs.
    """
    dataloader = get_split_dataloader(dm, split)

    smiles_out: List[str] = []
    feat_parts: List[np.ndarray] = []
    h_bound_parts: List[np.ndarray] = []
    c_bound_parts: List[np.ndarray] = []
    names: List[str] = []
    # Peaks are kept RAGGED (values concatenated + per-molecule counts) rather
    # than padded: 0.42 GB for this corpus instead of 2.4 GB, and no mask needed
    # to read them back. Boolean-indexing a [B, W] array is row-major, so the
    # flat values come out in molecule order already.
    h_val_parts: List[np.ndarray] = []
    h_nh_val_parts: List[np.ndarray] = []
    c_val_parts: List[np.ndarray] = []
    h_count_parts: List[np.ndarray] = []
    c_count_parts: List[np.ndarray] = []

    for batch in tqdm(dataloader, desc=f"  [{split}] peaks"):
        model_inputs, smiles = batch[0]
        cond = {k: v.detach().cpu().numpy() for k, v in model_inputs["condition"].items()}
        n = len(smiles)

        feats, batch_names = compute_spectral_feature_matrix(
            cond, np.arange(n), H_SHIFT_KEY, C_SHIFT_KEY, H_MASK_KEY, C_MASK_KEY,
            H_NH_KEY, None,
        )
        if feats is None:
            print(f"  [warn] condition dict lacks {H_SHIFT_KEY}/{C_SHIFT_KEY} -- "
                  "cannot build a spectral panel for this source.")
            return None
        names = batch_names

        # Truncation flags need the padded width, which is a property of this
        # source's own tensors and is gone once batches are concatenated.
        def at_bound(mask_key: str, shift_key: str) -> np.ndarray:
            arr = cond.get(mask_key)
            arr = np.isfinite(cond[shift_key]) if arr is None else np.asarray(arr).astype(bool)
            return arr.sum(axis=1) >= arr.shape[-1]

        smiles_out.extend(smiles)
        feat_parts.append(feats)
        h_bound_parts.append(at_bound(H_MASK_KEY, H_SHIFT_KEY))
        c_bound_parts.append(at_bound(C_MASK_KEY, C_SHIFT_KEY))

        h_mask = (np.asarray(cond[H_MASK_KEY]).astype(bool) if H_MASK_KEY in cond
                  else np.isfinite(cond[H_SHIFT_KEY]))
        c_mask = (np.asarray(cond[C_MASK_KEY]).astype(bool) if C_MASK_KEY in cond
                  else np.isfinite(cond[C_SHIFT_KEY]))
        h_val_parts.append(np.asarray(cond[H_SHIFT_KEY])[h_mask].astype(np.float32))
        c_val_parts.append(np.asarray(cond[C_SHIFT_KEY])[c_mask].astype(np.float32))
        h_nh_val_parts.append(np.asarray(cond[H_NH_KEY])[h_mask].astype(np.float32)
                               if H_NH_KEY in cond else np.zeros(int(h_mask.sum()), np.float32))
        h_count_parts.append(h_mask.sum(axis=1).astype(np.int64))
        c_count_parts.append(c_mask.sum(axis=1).astype(np.int64))

        if limit is not None and len(smiles_out) >= limit:
            break

    if not feat_parts:
        return None
    n_keep = len(smiles_out)
    h_counts = np.concatenate(h_count_parts)[:n_keep]
    c_counts = np.concatenate(c_count_parts)[:n_keep]
    # Trim the flat values to match, since `limit` may have cut the final batch.
    h_total = int(h_counts.sum())
    c_total = int(c_counts.sum())
    return {
        "smiles": smiles_out,
        "features": np.concatenate(feat_parts, axis=0)[:n_keep],
        "names": names,
        "h_at_bound": np.concatenate(h_bound_parts)[:n_keep],
        "c_at_bound": np.concatenate(c_bound_parts)[:n_keep],
        "h_values": np.concatenate(h_val_parts)[:h_total],
        "h_nh_values": np.concatenate(h_nh_val_parts)[:h_total],
        "c_values": np.concatenate(c_val_parts)[:c_total],
        "h_counts": h_counts,
        "c_counts": c_counts,
        "h_offsets": np.concatenate([[0], np.cumsum(h_counts)]).astype(np.int64),
        "c_offsets": np.concatenate([[0], np.cumsum(c_counts)]).astype(np.int64),
    }


def add_structure_features(spectrum_feats: np.ndarray, names: List[str], smiles: List[str],
                            h_at_bound: np.ndarray, c_at_bound: np.ndarray,
                            n_workers: int, h_reference: str) -> Tuple[np.ndarray, List[str]]:
    """Appends the 5 structure-derived features to a spectrum-only panel.

    Kept out of the batch loop so the RDKit work (AddHs + canonical ranking per
    molecule) runs once over the whole split in a worker pool.
    """
    worker = _StructureRef(h_reference)
    if n_workers <= 1:
        refs = [worker(s) for s in tqdm(smiles, desc="  structure refs")]
    else:
        with mp.Pool(n_workers) as pool:
            refs = list(tqdm(pool.imap(worker, smiles, chunksize=256), total=len(smiles),
                              desc=f"  structure refs ({n_workers} workers)"))

    nh_sum = spectrum_feats[:, names.index("H_nH_sum")]
    n_c_peaks = spectrum_feats[:, names.index("n_C_peaks")]
    extra = np.empty((len(smiles), len(QUALITY_FEATURE_NAMES) + len(STRUCTURE_FEATURE_NAMES)),
                     dtype=np.float64)
    for i, (n_h_ref, n_c_unique, n_c_total) in enumerate(refs):
        q = _quality_features(nh_sum[i], n_c_peaks[i], n_h_ref, n_c_unique,
                               bool(h_at_bound[i]), bool(c_at_bound[i]))
        extra[i] = [q[name] for name in QUALITY_FEATURE_NAMES] + [n_h_ref, n_c_total]

    return (np.hstack([spectrum_feats, extra]),
            list(names) + QUALITY_FEATURE_NAMES + STRUCTURE_FEATURE_NAMES)


class _StructureRef:
    """Picklable `_structure_reference` with the H reference bound, so the
    worker pool can carry the setting without a module-level global."""

    def __init__(self, h_reference: str):
        self.h_reference = h_reference

    def __call__(self, smi: str):
        return _structure_reference(smi, self.h_reference)


# -----------------------------------------------------------------------------
# Alignment to step 0's molecule order
# -----------------------------------------------------------------------------


def build_assignment(corpus_smiles: Sequence[str], corpus_dataset: Sequence[str],
                      extracted: Dict[str, Dict[str, Any]]
                      ) -> Tuple[List[Optional[Tuple[str, int]]], str, int]:
    """One (source, row) pointer per corpus molecule, in the corpus's own order.

    Computed once and reused for BOTH the feature matrix and the peak lists, so
    the two can never drift out of step with each other.

    Positional alignment is tried first (train/val are not subsampled, so the
    dataloader order should already agree) and falls back to a
    `(dataset, SMILES)` lookup. Unmatched molecules get `None` rather than
    shifting everything after them -- a misaligned panel is worse than a gappy
    one.
    """
    order = [name for name in dict.fromkeys(corpus_dataset) if name in extracted]
    positional_smiles = [s for name in order for s in extracted[name]["smiles"]]
    if len(positional_smiles) == len(corpus_smiles) and positional_smiles == list(corpus_smiles):
        assignment: List[Optional[Tuple[str, int]]] = []
        for name in order:
            assignment.extend((name, i) for i in range(len(extracted[name]["smiles"])))
        return assignment, "positional", len(assignment)

    lookup: Dict[Tuple[str, str], Tuple[str, int]] = {}
    for name, payload in extracted.items():
        for i, smi in enumerate(payload["smiles"]):
            lookup.setdefault((name, smi), (name, i))

    assignment = [lookup.get((ds, smi)) for ds, smi in zip(corpus_dataset, corpus_smiles)]
    return assignment, "smiles_join", sum(1 for a in assignment if a is not None)


def gather_features(assignment: Sequence[Optional[Tuple[str, int]]],
                     extracted: Dict[str, Dict[str, Any]], n_features: int) -> np.ndarray:
    out = np.full((len(assignment), n_features), np.nan, dtype=np.float64)
    for row, hit in enumerate(assignment):
        if hit is not None:
            out[row] = extracted[hit[0]]["features"][hit[1]]
    return out


def gather_ragged(assignment: Sequence[Optional[Tuple[str, int]]],
                   extracted: Dict[str, Dict[str, Any]], values_key: str, offsets_key: str,
                   extra_values_key: Optional[str] = None
                   ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Re-emit ragged peak arrays in the corpus's row order.

    Returns (values, offsets, extra_values) where `offsets` has len(assignment)+1
    entries, so molecule i's peaks are `values[offsets[i]:offsets[i+1]]`.
    Unmatched molecules get zero-length runs, which read back as empty arrays.
    """
    counts = np.zeros(len(assignment), dtype=np.int64)
    for row, hit in enumerate(assignment):
        if hit is not None:
            src, i = hit
            off = extracted[src][offsets_key]
            counts[row] = off[i + 1] - off[i]

    offsets = np.zeros(len(assignment) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    values = np.empty(int(offsets[-1]), dtype=np.float32)
    extra = np.empty(int(offsets[-1]), dtype=np.float32) if extra_values_key else None

    for row, hit in enumerate(assignment):
        if hit is None or counts[row] == 0:
            continue
        src, i = hit
        p = extracted[src]
        s0, s1 = p[offsets_key][i], p[offsets_key][i + 1]
        values[offsets[row]:offsets[row + 1]] = p[values_key][s0:s1]
        if extra is not None:
            extra[offsets[row]:offsets[row + 1]] = p[extra_values_key][s0:s1]
    return values, offsets, extra


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    args = parse_args()
    save_dir = args.save_dir or args.corpus_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    nmr3d_root = args.nmr3d_root.resolve()
    config_dir = str(nmr3d_root / "configs")
    manifest = read_corpus_manifest(args.corpus_dir, args.prefix)
    seed = int(manifest.get("seed", 1234))

    print("=" * 78)
    print("Spectral feature extraction (CPU-only; no model, no checkpoint forward pass)")
    print(f"  corpus dir  : {args.corpus_dir}")
    print(f"  configured from: {args.prefix}_manifest.json "
          f"(ckpt {Path(manifest.get('ckpt', '?')).name})")
    print(f"  condition   : {manifest['condition']}")
    print(f"  sources     : {[s['name'] for s in manifest['sources']]}")
    print(f"  splits      : {args.splits}")
    print(f"  H reference : {args.h_reference}")
    print("=" * 78)

    for split in args.splits:
        corpus_path = args.corpus_dir / f"{args.prefix}_{split}_global_cond.pt"
        if not corpus_path.exists():
            print(f"[warn] {corpus_path} not found -- skipping split '{split}'.")
            continue

        print(f"\n### split: {split} ###")
        if split == "test":
            print("  test cap lifted (config.yaml caps test at 100/source with a null seed, "
                  "so step 0's sample can only be recovered by SMILES).")

        extracted: Dict[str, Dict[str, Any]] = {}
        names: List[str] = []
        for source in manifest["sources"]:
            print(f"  -- source {source['name']} ({source['hydra_name']}) --")
            overrides = source_overrides(source) + test_split_overrides(split, seed)
            try:
                dm = build_datamodule(
                    config_dir, source["hydra_name"], source.get("split_suffix"),
                    manifest["ckpt_sigma"], manifest["condition"],
                    nmr3d_root=nmr3d_root, c_peak_norm_overrides=None,
                    extra_overrides=overrides,
                )
                payload = extract_source_split(dm, split, args.limit)
            except Exception as exc:
                print(f"  [warn] source '{source['name']}' failed: {exc}. Skipping.")
                continue
            if payload is None:
                continue
            extracted[source["name"]] = payload
            names = payload["names"]
            print(f"     {len(payload['smiles'])} molecules")

        if not extracted:
            print(f"[warn] no source yielded peaks for split '{split}' -- nothing saved.")
            continue

        # Structure-derived features, once per source over its whole SMILES list.
        full_names: List[str] = []
        for name, payload in extracted.items():
            feats_full, full_names = add_structure_features(
                payload["features"], names, payload["smiles"],
                payload["h_at_bound"], payload["c_at_bound"], args.n_workers,
                args.h_reference)
            payload["features"] = feats_full

        print(f"  loading {corpus_path.name} for its molecule order ...")
        corpus = torch.load(corpus_path, map_location="cpu")
        corpus_smiles, corpus_dataset = list(corpus["smiles"]), list(corpus["dataset"])
        n_corpus = len(corpus_smiles)
        del corpus

        assignment, method, n_matched = build_assignment(
            corpus_smiles, corpus_dataset, extracted)
        aligned = gather_features(assignment, extracted, len(full_names))
        h_values, h_offsets, h_nh_values = gather_ragged(
            assignment, extracted, "h_values", "h_offsets", extra_values_key="h_nh_values")
        c_values, c_offsets, _ = gather_ragged(
            assignment, extracted, "c_values", "c_offsets")
        print(f"  peaks: {len(h_values):,} 1H and {len(c_values):,} 13C values "
              f"({(h_values.nbytes + c_values.nbytes + h_nh_values.nbytes) / 1e6:.0f} MB)")
        n_missing = n_corpus - n_matched
        print(f"  alignment: {method}, matched {n_matched}/{n_corpus}"
              + (f" ({n_missing} rows left NaN)" if n_missing else " (complete)"))
        if n_missing:
            print("  [warn] unmatched rows are NaN, NOT dropped -- row i still corresponds to "
                  "embedding row i. Check --limit, or that this corpus came from these sources.")

        out_path = save_dir / f"{args.prefix}_{split}_spectral_features.pt"
        torch.save({
            "split": split,
            "prefix": args.prefix,
            "spectral_features": torch.from_numpy(aligned.astype(np.float32)),
            "spectral_feature_names": full_names,
            # Ragged peak lists: molecule i's 1H shifts are
            # h_peak_values[h_peak_offsets[i]:h_peak_offsets[i+1]], and likewise
            # for its nH integrations and 13C shifts. Stored this way rather
            # than padded (0.42 GB vs 2.4 GB here) and rather than as a list of
            # per-molecule arrays (which pickles slowly and carries millions of
            # objects). This is what lets the spectral-PCA analysis draw its traversal
            # filmstrips without a checkpoint.
            "h_peak_values": torch.from_numpy(h_values),
            "h_peak_nh_values": torch.from_numpy(h_nh_values),
            "h_peak_offsets": torch.from_numpy(h_offsets),
            "c_peak_values": torch.from_numpy(c_values),
            "c_peak_offsets": torch.from_numpy(c_offsets),
            "smiles": corpus_smiles,
            "dataset": corpus_dataset,
            "n_matched": int(n_matched),
            "n_missing": int(n_missing),
            "alignment": method,
            "h_reference": args.h_reference,
            "corpus_file": str(corpus_path),
            "corpus_manifest": {k: manifest[k] for k in
                                 ("ckpt", "ckpt_sigma", "condition", "sources",
                                  "c_peak_norm_overrides", "bounds_overrides")
                                 if k in manifest},
            "extracted_at": datetime.now(timezone.utc).isoformat(),
        }, out_path)
        print(f"✓ Saved {out_path} ({n_corpus} rows x {len(full_names)} features)")

    print("\nDone.")


if __name__ == "__main__":
    main()
