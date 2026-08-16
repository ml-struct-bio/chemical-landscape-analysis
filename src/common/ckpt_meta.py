"""ckpt_meta.py
===============

Reads the settings a checkpoint was actually TRAINED with straight out of its
Lightning ``hyper_parameters``, so extraction reproduces them exactly instead
of hardcoding a per-milestone constant that silently rots.

Two settings matter for embedding extraction, and they behave very
differently:

``diffusion_process_args.sigma_data`` -- the diffusion noise scale
------------------------------------------------------------------
This is restored automatically by ``load_from_checkpoint``; it is NOT read
from ``dataset_args.sigma_data`` at eval time. ``nmr-to-3d/src/main.py``
copies ``datamodule.sigma_data -> diffusion_process_args.sigma_data`` only at
TRAINING construction, and only when the diffusion value is still ``None``;
``model.save_hyperparameters()`` then freezes the resolved value into the
ckpt. ``NMRDataModule.sigma_data`` is read nowhere else and never reaches a
batch tensor.

So overriding ``dataset_args.sigma_data`` on the hydra CLI does NOT change
the model. Its one live effect is to short-circuit the datamodule's expensive
weighted-RMS recompute when a data yaml sets ``sigma_data: null`` (every
cotrain bundle does). We still set it -- to the checkpoint's own value -- so
that the number recorded in our manifests is the truth rather than a stale
constant. A wrong value here does not corrupt embeddings; it corrupts the
provenance record, which is how a wrong one survives unnoticed.

``dataset_args.c_peak_norm_args`` -- the 13C symmetry-merge normalization
-------------------------------------------------------------------------
This one DOES change the input tensors at extraction time. Per
``nmr-to-3d/configs/config.yaml``, it is deliberately a NORMALIZATION rather
than an augmentation: ``augment_c_peaks`` applies it even when
``augment=False``, so it reaches val/test/sampling too. Extracting with a
``dedup_p`` that differs from training feeds the model spectra it never saw,
with no error raised.

The cotrain-v3 checkpoint families differ in exactly this setting -- it is
what ``dedupOFF``/``dedupON`` in their directory names refers to:

    26-07-27-cotrain-v3-dedupOFF-s0    sigma 2.8418   dedup_p 0.0
    26-07-27-cotrain-v3-dedupON-s0     sigma 2.8418   dedup_p 1.0
    26-07-20-cotrain-v3-dedupOFF-s42   sigma 2.8418   dedup_p 0.0

Reading it from the ckpt removes the guesswork the extraction scripts' older
docstrings described as "genuinely uncertain for v3".

Cheap to call: uses ``torch.load(..., mmap=True)`` so a ~4.6 GB checkpoint is
not pulled into memory just to read its config. Needs only torch + omegaconf,
NOT nmr-to-3d on ``sys.path``, so it is safe to call outside
``nmr3d_import_scope``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

# Keys of the 13C normalization block, in the order the hydra overrides are emitted.
C_PEAK_NORM_KEYS = ("dedup_p", "dedup_tol", "dedup_mode")

# Two sigma values are considered "the same" below this absolute difference.
SIGMA_TOL = 1e-6


# Padding / tensor-shape bounds. In a cotrain run the datamodule takes these as
# the MAX across sources (see nmr-to-3d/src/data/datamodule.py's "max_n_atoms /
# h_max_n_peaks / c_max_n_peaks -> max (tensor-shape req)"), so every source was
# padded to the mixture-wide value, not to its own per-source yaml value.
# Re-extracting per-source would otherwise pad differently from training.
BOUNDS_KEYS = ("max_n_atoms", "h_max_n_peaks", "c_max_n_peaks", "h_max_nH")


@dataclass
class CkptMeta:
    """What a checkpoint says about its own training-time data handling."""
    path: Path
    sigma_data: Optional[float] = None
    c_peak_norm: Dict[str, Any] = field(default_factory=dict)
    max_n_atoms: Optional[int] = None
    h_max_n_peaks: Optional[int] = None
    c_max_n_peaks: Optional[int] = None
    h_max_nH: Optional[int] = None
    source_suffixes: Dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        srcs = ", ".join(f"{k}:{v}" for k, v in self.source_suffixes.items()) or "(single-source)"
        norm = ", ".join(f"{k}={self.c_peak_norm.get(k)}" for k in C_PEAK_NORM_KEYS
                         if k in self.c_peak_norm) or "(absent)"
        return (f"ckpt meta [{self.path.name}]\n"
                f"    sigma_data      : {self.sigma_data}\n"
                f"    c_peak_norm     : {norm}\n"
                f"    bounds          : max_n_atoms={self.max_n_atoms} "
                f"h/c peaks={self.h_max_n_peaks}/{self.c_max_n_peaks} "
                f"h_max_nH={self.h_max_nH}\n"
                f"    train sources   : {srcs}")

    def as_manifest_dict(self) -> Dict[str, Any]:
        return {
            "ckpt": str(self.path),
            "sigma_data": self.sigma_data,
            "c_peak_norm_args": dict(self.c_peak_norm),
            "max_n_atoms": self.max_n_atoms,
            "h_max_n_peaks": self.h_max_n_peaks,
            "c_max_n_peaks": self.c_max_n_peaks,
            "h_max_nH": self.h_max_nH,
            "train_source_suffixes": dict(self.source_suffixes),
        }


def _get(node: Any, *keys: str) -> Any:
    """Nested lookup that works for both plain dicts and OmegaConf nodes."""
    for key in keys:
        if node is None:
            return None
        try:
            if key not in node:
                return None
            node = node[key]
        except (TypeError, KeyError):
            return None
    return node


def _to_plain(node: Any) -> Any:
    """OmegaConf container -> plain python, leaving anything else untouched."""
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(node):
            return OmegaConf.to_container(node, resolve=True)
    except Exception:  # pragma: no cover - omegaconf always present in practice
        pass
    return node


def _load_hparams(ckpt: Path) -> Any:
    """Loads only the checkpoint's hyper_parameters. `mmap=True` keeps the
    ~4.6 GB of weights off the heap; older/legacy-format checkpoints that
    can't be mmapped fall back to a normal load."""
    try:
        blob = torch.load(str(ckpt), map_location="cpu", mmap=True, weights_only=False)
    except (RuntimeError, TypeError, ValueError):
        blob = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    hp = blob.get("hyper_parameters", {}) if isinstance(blob, dict) else {}
    # Lightning nests the hydra config under "cfg" for this project's modules.
    return hp.get("cfg", hp) if hasattr(hp, "get") else hp


def read_ckpt_meta(ckpt: Path) -> CkptMeta:
    """Reads sigma_data + c_peak_norm_args (and the padding bounds / train
    split suffixes, for logging) out of a checkpoint's saved hydra config.

    Fields the checkpoint doesn't carry come back as None / {} rather than
    raising, so an older-vintage ckpt degrades to "caller decides"."""
    ckpt = Path(ckpt)
    cfg = _load_hparams(ckpt)

    meta = CkptMeta(path=ckpt)
    sigma = _get(cfg, "diffusion_process_args", "sigma_data")
    meta.sigma_data = float(sigma) if sigma is not None else None

    norm = _to_plain(_get(cfg, "dataset_args", "c_peak_norm_args"))
    if isinstance(norm, dict):
        meta.c_peak_norm = {k: v for k, v in norm.items() if k in C_PEAK_NORM_KEYS}

    for attr in BOUNDS_KEYS:
        val = _get(cfg, "dataset_args", attr)
        if val is not None:
            setattr(meta, attr, int(val))

    sources = _to_plain(_get(cfg, "dataset_args", "sources")) or []
    if isinstance(sources, list):
        for s in sources:
            if isinstance(s, dict) and s.get("name"):
                meta.source_suffixes[s["name"]] = s.get("split_indices_suffix")

    return meta


def resolve_sigma_data(meta: CkptMeta, requested: Optional[float] = None,
                        *, strict: bool = False) -> Optional[float]:
    """Returns the sigma_data to use, which is always the checkpoint's own.

    `requested` is treated as an assertion, not an instruction: if it
    disagrees with the checkpoint we say so loudly and use the checkpoint's
    value anyway (or raise, with `strict=True`). This is the check that
    catches a sigma copy-pasted from a different milestone -- e.g. the May
    2026 cotrain mixture 2.8268 carried over to a v3 ckpt whose real value is
    2.8418."""
    if meta.sigma_data is None:
        if requested is None:
            raise ValueError(
                f"{meta.path.name} carries no diffusion_process_args.sigma_data and no "
                f"--ckpt-sigma was given; pass one explicitly.")
        print(f"[ckpt-meta] {meta.path.name} carries no sigma_data; using the supplied "
              f"{requested}.")
        return float(requested)

    if requested is not None and abs(float(requested) - meta.sigma_data) > SIGMA_TOL:
        msg = (f"--ckpt-sigma={requested} does NOT match the checkpoint's own "
               f"sigma_data={meta.sigma_data} ({meta.path.name}). The checkpoint's value "
               f"is authoritative -- the model restores it from its own hyper_parameters "
               f"regardless of what is passed here, so a stale value corrupts the "
               f"recorded provenance rather than the embeddings. Drop --ckpt-sigma and "
               f"let it be read from the checkpoint.")
        if strict:
            raise ValueError(msg)
        print(f"\n[ckpt-meta][WARNING] {msg}")
        print(f"[ckpt-meta] Proceeding with the checkpoint's {meta.sigma_data}.\n")

    return meta.sigma_data


def c_peak_norm_overrides(meta: CkptMeta, forced: Optional[Dict[str, Any]] = None) -> List[str]:
    """Hydra override strings pinning `dataset_args.c_peak_norm_args` to what
    the checkpoint trained with (or to `forced`, if given).

    Emitted for every dataset composed during extraction, matching the fact
    that `c_peak_norm_args` is a single global `dataset_args` block, not a
    per-source one. Returns [] when the checkpoint predates the block, which
    leaves the hydra config's own default in place."""
    values = forced if forced is not None else meta.c_peak_norm
    if not values:
        return []
    return [f"dataset_args.c_peak_norm_args.{k}={values[k]}"
            for k in C_PEAK_NORM_KEYS if k in values]


def bounds_overrides(meta: CkptMeta) -> List[str]:
    """Hydra overrides pinning the padding / tensor-shape bounds to the
    MIXTURE-WIDE values the checkpoint trained with.

    Needed because extraction composes one dataset config per source, while
    cotrain training composed a single config whose bounds were the max across
    all sources. Left alone, spectranp would be padded to its own 47/98 peaks
    and uspto to 28/74 rather than the 61/116 they actually saw in training.
    Every source is padded UP to the mixture value here, never truncated.

    `h_max_nH` is included for exactness but is inert at extraction time: it
    only sizes the nH embedding table at model-construction time, and
    `load_from_checkpoint` restores that table from the checkpoint (max_nH=20
    for the cotrain-v3 ckpts, regardless of the dataset_args value).

    NOT applied when embedding held-out datasets: those are
    by definition outside the training mixture, and forcing e.g. c_max_n_peaks
    down to 116 would TRUNCATE real peaks on datasets that carry more
    (real-npmrd-lq declares 198)."""
    out = []
    for key in BOUNDS_KEYS:
        val = getattr(meta, key, None)
        if val is not None:
            out.append(f"dataset_args.{key}={val}")
    return out
