"""Loading the extracted corpus, and choosing which embedding to analyze.

Three extraction outputs feed the analyses:

    <prefix>_<split>_global_cond.pt        encoder embedding, SMILES, dataset, ECFP
    <prefix>_<split>_layerwise.pt          decoder trunk hidden states per (layer, timestep)
    <prefix>_<split>_spectral_features.pt  NMR feature panel + raw peak lists

The spectral panel is written in `global_cond`'s own molecule order, so pairing
those two is positional. The layerwise file is not: it is subsampled at
extraction time (`--n-samples-per-dataset`), so a decoder embedding has to be
*joined* back to the corpus order before it can be paired with spectra. That
join is the fiddly part of this module.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from src.common.paths import cache_dir


# -----------------------------------------------------------------------------
# Which embedding to run the PCA over
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class EmbeddingSpec:
    """One embedding space to analyze.

    `kind` is "encoder" (the peak embedder's pooled `global_cond`), "ecfp" (the
    Morgan fingerprint, a model-free structural baseline), or "decoder" (a trunk
    hidden state). Decoder specs carry the stream, transformer layer and
    diffusion timestep identifying one entry of the layerwise file's
    `layer_timestep_data`.
    """
    kind: str
    stream: Optional[str] = None      # "x_hidden_mean" | "y_hidden_mean"
    layer: Optional[int] = None
    timestep: Optional[float] = None

    @property
    def slug(self) -> str:
        """The `data/`/`figures/` tag. Stable and filesystem-safe."""
        if self.kind == "encoder":
            return "global_cond"
        if self.kind == "ecfp":
            return "ecfp"
        short = "x" if self.stream == "x_hidden_mean" else "y"
        return f"decoder_{short}_L{self.layer:02d}_t{self.timestep:g}"

    @property
    def label(self) -> str:
        """Human-readable, for figure titles and log lines."""
        if self.kind == "encoder":
            return "global_cond (encoder)"
        if self.kind == "ecfp":
            return "ECFP fingerprint"
        stream = "atom/coord stream" if self.stream == "x_hidden_mean" else "NMR stream"
        return f"decoder layer {self.layer}, t={self.timestep:g} ({stream})"

    @property
    def cache_key(self) -> str:
        """`embedding_key` as the shared UMAP cache fingerprints it.

        Encoder and ECFP keep the previous pipeline's keys verbatim, so this
        repo still hits its cached full-corpus fits -- those cost hours.

        Decoder specs carry their own stream/layer/timestep, which the previous
        pipeline did NOT: it keyed every decoder fit as the bare string
        "decoder", and every other fingerprinted field (n_neighbors, min_dist,
        metric, pca_dim, scale, seed, n_points, n_dims) is identical across
        layers of one extraction. All 36 (stream, layer, timestep) combos
        therefore collided on a single hash -- whichever fitted first won, and
        every other layer silently loaded THAT projection instead of fitting its
        own. The old mitigation was "don't point two layers at one decoder cache
        directory", which cannot survive `--embeddings all-decoder`: it loops
        every layer through the one default directory, and the shape check in
        `project_and_store` cannot catch it because the shapes match exactly.

        Decoder entries written under the old key are unreachable from here.
        That is deliberate -- they cannot be attributed to a layer, so there is
        nothing safe to reuse them for.
        """
        if self.kind == "encoder":
            return "global_cond"
        if self.kind == "ecfp":
            return "ecfp"
        return self.slug  # decoder_<stream>_L<layer>_t<timestep>, i.e. per-fit


_STREAM_ALIASES = {"x": "x_hidden_mean", "y": "y_hidden_mean",
                   "x_hidden_mean": "x_hidden_mean", "y_hidden_mean": "y_hidden_mean"}


def available_decoder_keys(layerwise_dir: Path, prefix: str,
                           splits: Sequence[str]) -> List[Tuple[str, int, float]]:
    """Every (stream, layer, timestep) present in the layerwise file, sorted.

    Read from the first split that exists -- the extraction writes the same
    layer/timestep grid for all of them.
    """
    for split in splits:
        path = Path(layerwise_dir) / f"{prefix}_{split}_layerwise.pt"
        if not path.exists():
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        out = []
        for (layer, timestep), streams in payload["layer_timestep_data"].items():
            for stream, arr in streams.items():
                if arr is not None:
                    out.append((stream, int(layer), float(timestep)))
        return sorted(out, key=lambda t: (t[0], t[2], t[1]))
    raise FileNotFoundError(
        f"No {prefix}_<split>_layerwise.pt found in {layerwise_dir} for splits {list(splits)}. "
        f"Run extraction/02_decoder_layers.py first.")


def parse_embedding_specs(tokens: Sequence[str], layerwise_dir: Path, prefix: str,
                          splits: Sequence[str]) -> List[EmbeddingSpec]:
    """Turns CLI tokens into specs.

    Accepted forms::

        global_cond            the encoder embedding
        x:5@0.001              decoder x-stream, layer 5, timestep 0.001
        y:11@1.0               decoder y-stream, layer 11, timestep 1.0
        x:5                    ... first timestep available for that layer
        all-decoder            every (stream, layer, timestep) in the file
        all                    global_cond + all-decoder

    Only the forms that mention a decoder read the layerwise file, so an
    encoder-only run never needs one to exist.
    """
    specs: List[EmbeddingSpec] = []
    keys: Optional[List[Tuple[str, int, float]]] = None

    def load_keys() -> List[Tuple[str, int, float]]:
        nonlocal keys
        if keys is None:
            keys = available_decoder_keys(layerwise_dir, prefix, splits)
        return keys

    for token in tokens:
        low = token.strip().lower()
        if low in ("global_cond", "encoder"):
            specs.append(EmbeddingSpec(kind="encoder"))
            continue
        if low == "ecfp":
            specs.append(EmbeddingSpec(kind="ecfp"))
            continue
        if low in ("all", "all-decoder", "all_decoder"):
            if low == "all":
                specs.append(EmbeddingSpec(kind="encoder"))
                specs.append(EmbeddingSpec(kind="ecfp"))
            specs.extend(EmbeddingSpec(kind="decoder", stream=s, layer=l, timestep=t)
                         for s, l, t in load_keys())
            continue

        if ":" not in low:
            raise ValueError(
                f"Unrecognized --embeddings entry {token!r}. Expected 'global_cond', "
                f"'all', 'all-decoder', or a decoder spec like 'x:5@0.001'.")
        stream_tok, rest = low.split(":", 1)
        if stream_tok not in _STREAM_ALIASES:
            raise ValueError(f"Unknown stream {stream_tok!r} in {token!r}; expected 'x' or 'y'.")
        stream = _STREAM_ALIASES[stream_tok]

        if "@" in rest:
            layer_tok, ts_tok = rest.split("@", 1)
            layer, timestep = int(layer_tok), float(ts_tok)
            if (stream, layer, timestep) not in load_keys():
                raise ValueError(
                    f"{token!r} is not in the layerwise file. Available: "
                    + ", ".join(f"{s[0]}:{l}@{t:g}" for s, l, t in load_keys()[:12]) + " ...")
            specs.append(EmbeddingSpec(kind="decoder", stream=stream, layer=layer, timestep=timestep))
        else:
            layer = int(rest)
            matches = sorted(t for s, l, t in load_keys() if s == stream and l == layer)
            if not matches:
                raise ValueError(f"No timesteps for layer {layer}, stream {stream_tok} in the "
                                 f"layerwise file.")
            specs.append(EmbeddingSpec(kind="decoder", stream=stream, layer=layer,
                                       timestep=matches[0]))

    # De-duplicate while preserving order -- `all` plus an explicit spec is a
    # natural thing to type and should not analyze the same embedding twice.
    seen, unique = set(), []
    for spec in specs:
        if spec.slug not in seen:
            seen.add(spec.slug)
            unique.append(spec)
    return unique


# -----------------------------------------------------------------------------
# Ragged peak storage
# -----------------------------------------------------------------------------


class RaggedPeaks:
    """Indexable view over `extraction/01`'s flat values + offsets storage.

    Peaks are stored flat rather than as millions of small arrays (0.42 GB vs
    2.4 GB on this corpus). A traversal only ever touches a few dozen molecules,
    so slicing on access beats materializing every list up front.
    """

    __slots__ = ("values", "offsets")

    def __init__(self, values: np.ndarray, offsets: np.ndarray):
        self.values = np.asarray(values)
        self.offsets = np.asarray(offsets)

    def __len__(self) -> int:
        return max(len(self.offsets) - 1, 0)

    def __getitem__(self, i: int) -> np.ndarray:
        return self.values[self.offsets[i]:self.offsets[i + 1]]

    def take(self, indices: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
        """(flat values, offsets) for a subset -- how the step molecules' peaks
        get written into `data/` without carrying the whole corpus along."""
        parts = [np.asarray(self[int(i)]) for i in indices]
        offsets = np.zeros(len(parts) + 1, dtype=np.int64)
        if parts:
            offsets[1:] = np.cumsum([len(p) for p in parts])
            return np.concatenate(parts).astype(np.float32), offsets
        return np.zeros(0, dtype=np.float32), offsets


def _concat_ragged(parts: Sequence[Optional[RaggedPeaks]]) -> Optional[RaggedPeaks]:
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    values = np.concatenate([p.values for p in parts])
    offsets = [np.asarray([0], dtype=np.int64)]
    base = 0
    for p in parts:
        offsets.append(np.asarray(p.offsets[1:]) + base)
        base += int(p.offsets[-1])
    return RaggedPeaks(values, np.concatenate(offsets))


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------


def load_encoder_corpus(data_dir: Path, spectral_dir: Path, prefix: str,
                        splits: Sequence[str], *,
                        embedding_keys: Sequence[str] = ("global_cond",),
                        require_spectral: bool = True,
                        want_smiles: bool = True) -> Dict[str, Any]:
    """`global_cond` (and/or `ecfp`) paired with the spectral panel and peaks.

    `embedding_keys` selects which tensors to materialize -- these are 13 GB and
    20 GB respectively on the full train split, so loading both when only one is
    wanted is the difference between a comfortable job and an OOM.

    `require_spectral=False` lets callers that only colour by structure (no
    spectral features) proceed without `extraction/01`'s output.

    Alignment between the two files is *verified*, not assumed: `extraction/01`
    stores the SMILES list it aligned against, so a mismatch means the files
    describe different corpora and every correlation drawn from them would be
    meaningless. A split that fails the check is skipped with a warning.
    """
    smiles: List[str] = []
    dataset: List[str] = []
    split_tag: List[str] = []
    emb_parts: Dict[str, List[np.ndarray]] = {k: [] for k in embedding_keys}
    feat_parts: List[np.ndarray] = []
    h_parts: List[Optional[RaggedPeaks]] = []
    c_parts: List[Optional[RaggedPeaks]] = []
    nh_parts: List[Optional[RaggedPeaks]] = []
    feature_names: List[str] = []
    n_loaded = 0

    for split in splits:
        emb_path = Path(data_dir) / f"{prefix}_{split}_global_cond.pt"
        spec_path = Path(spectral_dir) / f"{prefix}_{split}_spectral_features.pt"
        if not emb_path.exists():
            print(f"[warn] {emb_path} not found -- skipping split {split!r}.")
            continue
        have_spectral = spec_path.exists()
        if not have_spectral:
            if require_spectral:
                print(f"[warn] {spec_path} not found -- skipping split {split!r}. "
                      "Run extraction/01_spectral_features.py to create it.")
                continue
            print(f"  [note] {spec_path.name} not found; continuing without spectral "
                  f"features for split {split!r}.")

        print(f"  loading {split}: {emb_path.name}"
              + (f" + {spec_path.name}" if have_spectral else ""))
        emb_payload = torch.load(emb_path, map_location="cpu", weights_only=False)
        split_smiles = list(emb_payload["smiles"])

        if have_spectral:
            spec_payload = torch.load(spec_path, map_location="cpu", weights_only=False)
            features = np.asarray(spec_payload["spectral_features"], dtype=np.float64)
            names = list(spec_payload["spectral_feature_names"])

            if len(features) != len(split_smiles):
                print(f"[warn] {spec_path.name}: {len(features)} feature rows vs "
                      f"{len(split_smiles)} molecules -- skipping split {split!r}.")
                continue
            saved_smiles = list(spec_payload.get("smiles", split_smiles))
            if saved_smiles != split_smiles:
                n_bad = sum(1 for a, b in zip(saved_smiles, split_smiles) if a != b)
                print(f"[warn] {spec_path.name}: SMILES disagree with the embeddings at "
                      f"{n_bad}/{len(split_smiles)} positions -- skipping split {split!r}.")
                continue
            if feature_names and names != feature_names:
                print(f"[warn] {spec_path.name}: feature panel differs from earlier splits "
                      f"({len(names)} vs {len(feature_names)}) -- skipping split {split!r}.")
                continue
            feature_names = names
            feat_parts.append(features)

        n_loaded += 1
        if want_smiles:
            smiles.extend(split_smiles)
        dataset.extend(list(emb_payload["dataset"]))
        split_tag.extend([split] * len(split_smiles))
        for key in embedding_keys:
            if key not in emb_payload:
                raise SystemExit(f"{emb_path.name} has no '{key}' tensor. "
                                 f"Keys: {sorted(emb_payload)}")
            emb_parts[key].append(emb_payload[key].numpy().astype(np.float32))
        del emb_payload

        if have_spectral and all(k in spec_payload for k in ("h_peak_values", "h_peak_offsets",
                                                             "c_peak_values", "c_peak_offsets")):
            h_parts.append(RaggedPeaks(np.asarray(spec_payload["h_peak_values"]),
                                       np.asarray(spec_payload["h_peak_offsets"])))
            c_parts.append(RaggedPeaks(np.asarray(spec_payload["c_peak_values"]),
                                       np.asarray(spec_payload["c_peak_offsets"])))
            nh_parts.append(RaggedPeaks(np.asarray(spec_payload["h_peak_nh_values"]),
                                        np.asarray(spec_payload["h_peak_offsets"]))
                            if "h_peak_nh_values" in spec_payload else None)
        elif have_spectral:
            h_parts.append(None)
        if have_spectral and int(spec_payload.get("n_missing", 0)):
            print(f"    {spec_payload['n_missing']} rows are NaN (unmatched during alignment)")

    if n_loaded == 0:
        raise SystemExit(
            "No split could be loaded -- nothing to analyze.\n"
            f"Looked for {prefix}_<split>_global_cond.pt in {data_dir}\n"
            f"        and {prefix}_<split>_spectral_features.pt in {spectral_dir}")

    n_rows = len(dataset)
    have_peaks = len(h_parts) == n_loaded and bool(h_parts) and all(p is not None for p in h_parts)
    h_peaks = _concat_ragged(h_parts) if have_peaks else None
    c_peaks = _concat_ragged(c_parts) if have_peaks else None
    h_nh = _concat_ragged(nh_parts) if have_peaks and all(p is not None for p in nh_parts) else None
    if have_peaks and h_peaks is not None and len(h_peaks) != n_rows:
        print(f"[warn] peak lists cover {len(h_peaks)} molecules but the corpus has "
              f"{n_rows} -- dropping peaks, so filmstrips will be skipped.")
        h_peaks = c_peaks = h_nh = None

    embeddings = {k: np.concatenate(v, axis=0).astype(np.float32)
                  for k, v in emb_parts.items() if v}
    result = {
        "smiles": smiles,
        "dataset": np.asarray(dataset),
        "split": np.asarray(split_tag),
        "embeddings": embeddings,
        "spectral_features": (np.concatenate(feat_parts, axis=0) if feat_parts
                              else np.zeros((n_rows, 0))),
        "spectral_feature_names": feature_names,
        "h_peaks": h_peaks,
        "c_peaks": c_peaks,
        "h_nh": h_nh,
    }
    # Back-compat for callers that predate multi-key loading (04_joint_pca).
    if "global_cond" in embeddings:
        result["embedding"] = embeddings["global_cond"]
    return result


def load_layerwise(layerwise_dir: Path, prefix: str, splits: Sequence[str],
                   wanted: Sequence[EmbeddingSpec]) -> Dict[str, Any]:
    """Decoder hidden states for exactly the (stream, layer, timestep) combos in
    `wanted`, concatenated across splits.

    Only the requested combos are kept. That matters: the layerwise file holds
    up to 72 embeddings of the same molecules, so materializing all of them to
    analyze one would cost tens of GB for no reason.
    """
    keep = {(s.stream, s.layer, s.timestep) for s in wanted if s.kind == "decoder"}
    smiles: List[str] = []
    dataset: List[str] = []
    mol_idx_parts: List[Optional[np.ndarray]] = []
    blocks: Dict[Tuple[str, int, float], List[np.ndarray]] = {k: [] for k in keep}

    for split in splits:
        path = Path(layerwise_dir) / f"{prefix}_{split}_layerwise.pt"
        if not path.exists():
            print(f"[warn] {path} not found -- skipping split {split!r}.")
            continue
        print(f"  loading {split}: {path.name}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        smiles.extend(payload["smiles"])
        dataset.extend(payload["dataset"])
        mol_idx_parts.append(payload.get("mol_idx"))
        for (layer, timestep), streams in payload["layer_timestep_data"].items():
            for stream, arr in streams.items():
                key = (stream, int(layer), float(timestep))
                if key in blocks and arr is not None:
                    blocks[key].append(np.asarray(arr, dtype=np.float32))

    if not smiles:
        raise SystemExit(
            f"No {prefix}_<split>_layerwise.pt could be loaded from {layerwise_dir} "
            f"for splits {list(splits)}.\nRun extraction/02_decoder_layers.py first.")

    mol_idx = None
    if mol_idx_parts and all(p is not None for p in mol_idx_parts):
        mol_idx = np.concatenate([np.asarray(p) for p in mol_idx_parts])

    embeddings = {k: np.concatenate(v, axis=0) for k, v in blocks.items() if v}
    missing = keep - set(embeddings)
    if missing:
        raise SystemExit(f"Requested decoder embeddings absent from the layerwise file: {missing}")

    return {"smiles": smiles, "dataset": np.asarray(dataset), "mol_idx": mol_idx,
            "embeddings": embeddings}


def _canonical_worker(smi: str) -> Optional[str]:
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol is not None else None


def _canonicalize_many(smiles: Sequence[str], n_workers: int = 1) -> List[Optional[str]]:
    if n_workers <= 1:
        return [_canonical_worker(s) for s in smiles]
    import multiprocessing as mp
    with mp.Pool(n_workers) as pool:
        return list(pool.imap(_canonical_worker, smiles, chunksize=1024))


def align_to_corpus(query_smiles: Sequence[str], query_mol_idx: Optional[np.ndarray],
                    ref_smiles: Sequence[str], ref_mol_idx: Optional[np.ndarray],
                    n_workers: int = 1) -> np.ndarray:
    """For each query row, the matching index into the reference (-1 if none).

    Three strategies, cheapest first:

    1. `mol_idx` on both sides -- the join key `extraction/02` saves precisely
       so this is exact. In practice it is often None (the current corpus has
       it unset), hence the fallbacks.
    2. Raw SMILES string equality. Both files come from the same datamodule, so
       the strings are usually identical and this resolves nearly everything
       for the cost of a dict.
    3. Canonical SMILES, computed only for the rows step 2 missed. RDKit
       canonicalization over millions of reference molecules is expensive, so
       it is worth avoiding when steps 1-2 already answered.
    """
    if query_mol_idx is not None and ref_mol_idx is not None:
        ref_map = {int(v): i for i, v in enumerate(ref_mol_idx)}
        return np.array([ref_map.get(int(v), -1) for v in query_mol_idx], dtype=np.int64)

    raw_map: Dict[str, int] = {}
    for i, s in enumerate(ref_smiles):
        raw_map.setdefault(s, i)
    out = np.array([raw_map.get(s, -1) for s in query_smiles], dtype=np.int64)

    unmatched = np.flatnonzero(out < 0)
    if len(unmatched):
        print(f"  {len(unmatched)}/{len(out)} rows unmatched on raw SMILES; "
              f"falling back to canonical SMILES for those.")
        ref_canon = _canonicalize_many(list(ref_smiles), n_workers)
        canon_map: Dict[str, int] = {}
        for i, c in enumerate(ref_canon):
            if c is not None:
                canon_map.setdefault(c, i)
        q_canon = _canonicalize_many([query_smiles[i] for i in unmatched], n_workers)
        for pos, c in zip(unmatched, q_canon):
            if c is not None:
                out[pos] = canon_map.get(c, -1)
    return out


_LOW_MATCH_FRACTION = 0.5


def _explain_bad_join(n_matched: int, n_total: int, splits: Sequence[str]) -> str:
    msg = (f"Only {n_matched}/{n_total} decoder molecules matched the global_cond corpus.\n"
           "The layerwise and global_cond files describe different molecule sets, so any "
           "spectral feature or descriptor joined onto them would belong to the wrong "
           "molecule.\n")
    if "test" in splits:
        msg += (
            "\nMost likely cause: the `test` split.\n"
            "  nmr-to-3d/configs/config.yaml sets dataset_args.test_args.test_samples=100 with "
            "a null test_seed, and NMRDataModule.setup('test') applies it unconditionally. "
            "Neither extraction script overrides it, so each drew its OWN random 100 molecules "
            "per source with default_rng(None) -- the two test splits are simply different "
            "samples, and their overlap is chance.\n"
            "  Use --splits train (or val) for decoder embeddings, or re-extract both files "
            "with the cap lifted and a pinned test_seed.\n")
    else:
        msg += ("\nCheck that both files came from the same extraction run against the same "
                "checkpoint and the same --splits.\n")
    return msg


def load_decoder_corpus(data_dir: Path, layerwise_dir: Path, spectral_dir: Path, prefix: str,
                        splits: Sequence[str], specs: Sequence[EmbeddingSpec],
                        n_workers: int = 1, *, require_spectral: bool = True,
                        embedding_keys: Sequence[str] = ("global_cond",)) -> Dict[str, Any]:
    """Decoder embeddings joined to the spectral panel (and, via `embedding_keys`,
    to any baseline encoder-side embeddings -- e.g. `ecfp` -- gathered through
    the same join).

    The layerwise extraction is subsampled, so its molecules are a subset of the
    `global_cond` corpus in a different order. Everything the analysis needs
    besides the embedding itself -- SMILES, dataset labels, spectral features,
    peak lists, baseline embeddings -- is indexed out of the encoder corpus
    through that join, and rows that fail to match are dropped (they would
    otherwise carry NaN spectra into every correlation).

    `require_spectral=False` is for analyses that never touch the spectral
    panel (e.g. layer-comparison metrics): it skips the `extraction/01`
    dependency entirely rather than silently forcing it on every decoder-corpus
    caller, most of which do want spectra for filmstrips/colouring.
    """
    layerwise = load_layerwise(layerwise_dir, prefix, splits, specs)
    encoder = load_encoder_corpus(data_dir, spectral_dir, prefix, splits,
                                  embedding_keys=embedding_keys, require_spectral=require_spectral)

    print(f"  joining {len(layerwise['smiles'])} decoder rows onto "
          f"{len(encoder['smiles'])} corpus rows")
    idx = align_to_corpus(layerwise["smiles"], layerwise["mol_idx"],
                          encoder["smiles"], None, n_workers=n_workers)
    keep = np.flatnonzero(idx >= 0)
    # A partial join is not a warning-level event. Everything except the decoder
    # embedding itself -- SMILES, spectra, descriptors -- is indexed through it,
    # so a mostly-failed join yields a small, plausible-looking, wrong analysis
    # rather than an obvious failure. Refuse instead.
    if len(keep) < _LOW_MATCH_FRACTION * len(idx):
        raise SystemExit(_explain_bad_join(len(keep), len(idx), splits))
    if len(keep) < len(idx):
        print(f"[warn] {len(idx) - len(keep)}/{len(idx)} decoder rows had no match in the "
              f"corpus and were dropped.")
    ref = idx[keep]

    return {
        "smiles": [encoder["smiles"][i] for i in ref],
        "dataset": encoder["dataset"][ref],
        "embeddings": {k: v[keep] for k, v in layerwise["embeddings"].items()},
        "baseline_embeddings": {k: v[ref] for k, v in encoder["embeddings"].items()},
        "spectral_features": encoder["spectral_features"][ref],
        "spectral_feature_names": encoder["spectral_feature_names"],
        # Peaks stay in their corpus-order ragged store; `ref` maps a decoder
        # row to the corpus row whose peaks it should draw.
        "h_peaks": encoder["h_peaks"],
        "c_peaks": encoder["c_peaks"],
        "h_nh": encoder["h_nh"],
        "peak_index": ref,
    }


# -----------------------------------------------------------------------------
# Descriptor cache
# -----------------------------------------------------------------------------


def cached_descriptor_matrix(smiles: Sequence[str], descriptor_names: List[str],
                             n_workers: int = 1, use_cache: bool = True) -> np.ndarray:
    """`compute_descriptor_matrix`, memoized on disk under `cache/descriptors/`.

    The RDKit pass dominates a full-corpus run and depends only on the molecule
    set, not on which embedding is being analyzed. Running all 72 decoder
    embeddings would otherwise repeat one multi-million-molecule pass 72 times.

    Analysis-only, by construction: `cache/` is never read by plotting.
    """
    from src.analysis.descriptors import compute_descriptor_matrix

    if not use_cache:
        return compute_descriptor_matrix(smiles, descriptor_names, n_workers)

    digest = hashlib.sha256()
    digest.update(json.dumps(list(descriptor_names)).encode())
    digest.update(str(len(smiles)).encode())
    # Hashing every SMILES is itself a full pass, but a cheap one next to RDKit
    # parsing, and it is what makes a stale cache impossible rather than merely
    # unlikely.
    for smi in smiles:
        digest.update(smi.encode("utf-8", "replace"))
        digest.update(b"\0")
    key = digest.hexdigest()[:32]

    path = cache_dir("descriptors", create=True) / f"{key}.npy"
    if path.exists():
        print(f"  reusing cached descriptor matrix {path.name}")
        return np.load(path)

    matrix = compute_descriptor_matrix(smiles, descriptor_names, n_workers)
    np.save(path, matrix)
    print(f"  cached descriptor matrix -> {path}")
    return matrix
