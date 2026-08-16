#!/usr/bin/env python
"""
02_decoder_layers.py
==========================================

Variant of `extraction/00_global_cond.py` that additionally extracts
**decoder (trunk) hidden states** at a user-specified set of transformer
layers and diffusion timesteps -- not just the peak-embedder's pooled
`global_cond` -- following the noising/pooling recipe from an earlier
one-off multilayer extraction script.

For each (layer, timestep) combination, this pulls:
    - `x_hidden_mean`: mean-pooled (over valid atoms) trunk hidden state
      for the atom/coordinate stream, at that layer and noise level.
    - `y_hidden_mean`: mean-pooled (over valid NMR tokens) trunk hidden
      state for the NMR-conditioning stream, at that layer and noise level.

Noising matches the diffusion training process: at t<=0.001 the "clean"
coordinates are used directly; at higher t, coordinates are interpolated
toward Gaussian noise via the model's own `interpolate` (flow-matching)
method and re-centered, exactly mirroring what the model sees during
training/sampling at that noise level.

This is *far* more expensive per batch than plain peak-embedder extraction
(one trunk forward pass per (layer, timestep) pair, i.e.
`len(layers) * len(timesteps)` forward passes per batch), so unlike
`extraction/00_global_cond.py` this one defaults to a per-source-dataset
sample cap (`--n-samples-per-dataset`, matching an earlier reference
script's `N_SAMPLES=5000`) rather than processing everything.

Lean by default: no duplicated global_cond/ECFP
--------------------------------------------------
`global_cond` doesn't depend on layer/timestep, and ECFP doesn't depend on
the model at all -- both are already saved once per molecule by
`extraction/00_global_cond.py`. This script saves `mol_idx` specifically so
you can join back to that file for either field rather than paying to
duplicate potentially-huge tensors under every (layer, timestep) key. Pass
`--save-global-cond` / `--save-ecfp` if you want a self-contained file
anyway (e.g. for a one-off run where you don't have the paired
global_cond extraction handy).

Version lineage: original -> clean/clean-v2 (2026-07-17) -> v3 (2026-07-20)
------------------------------------------------------------------------------
Same underlying dataset-composition history as `extraction/00_global_cond.py`
(see that script's docstring for the full account). This version targets
the 2026-07-20 "v3" split rebuild (`cotrain-nmrexp-spectranp-uspto-v3.yaml`),
composed via the `-v3` single-source configs so the extracted corpus matches
the training mixture exactly.

Concretely, vs. the clean-v2 `DEFAULT_SOURCES`, BOTH the hydra config and the
suffix moved to their v3 counterparts:
    - nmrexp:    `nmrexp`         `_nmrpeak_full_realdedup` -> `nmrexp-v3`    `_nmrpeak_full_v3_realdedup`
    - spectranp: `spectranp-760k` `_or_le200_realdedup`     -> `spectranp-v3` `_or_v3base_realdedup`
                 (valid_h_key=valid_indices_h_clean still applies; the v3 yaml sets it too)
    - uspto:     `uspto`          `_both_full_realdedup`    -> `uspto-v3`     `_both_full_realdedup_v3`

Using the `-v3` configs (rather than the base ones with a suffix override) is
what makes the extracted corpus match the training mixture exactly: verified
field-by-field against 26-07-27-cotrain-v3-dedupOFF-s0/epoch1399's own
`dataset_args`. See the DEFAULT_SOURCES comment block below.

Expected v3 train/val/test counts (from the v3 yaml's own comments), useful
for sanity-checking a --n-samples-per-dataset-capped smoke test's printed
per-source line:
    nmrexp:    train=1,187,838 (val=56,082,  test=123,801)
    spectranp: train=674,761   (val=34,226,  test=35,129)
    uspto:     train=662,342   (val=34,730,  test=78,645)

`c_peak_norm_args` (13C symmetry-merge) -- READ FROM THE CHECKPOINT
--------------------------------------------------------------------------
This is the setting that genuinely changes what gets extracted: per
`nmr-to-3d/configs/config.yaml` it is a NORMALIZATION, not an augmentation,
so `augment_c_peaks` applies it at val/test too. A `dedup_p` that differs
from training feeds the model spectra it never saw, silently.

No longer guessed per version -- it is read off the checkpoint's own
hyper_parameters and pinned on every source (it is a single GLOBAL
`dataset_args` block, not per-source). That is also exactly what the
cotrain-v3 `dedupOFF`/`dedupON` families differ in:

    26-07-27-cotrain-v3-dedupOFF-s0    sigma 2.8418   dedup_p 0.0
    26-07-27-cotrain-v3-dedupON-s0     sigma 2.8418   dedup_p 1.0
    26-07-20-cotrain-v3-dedupOFF-s42   sigma 2.8418   dedup_p 0.0

`--apply-cpeak-norm-overrides` is deprecated (errors if it contradicts the
checkpoint); `--no-ckpt-cpeak-norm` falls back to the hydra config defaults.

`--ckpt-sigma` is optional and does NOT scale the model
--------------------------------------------------------------------------
An earlier revision of this docstring implied a stale sigma would corrupt the
embeddings. It does not. The model restores
`diffusion_process_args.sigma_data` from its own checkpoint hyper_parameters;
what this flag sets is `dataset_args.sigma_data` -> `NMRDataModule.sigma_data`,
which no forward pass reads. Its only live effect is skipping the datamodule's
weighted-RMS recompute for `sigma_data: null` configs -- so a wrong value
corrupts the recorded provenance rather than the data, which is precisely how
one survives unnoticed. The value is now read from the checkpoint (2.8418 for
every cotrain-v3 ckpt); passing `--ckpt-sigma` only asserts it. See
`src/common/ckpt_meta.py`.

`--dedup` requires an explicit `dedup_suffix` configured per source (via
--sources-json); naively appending "_dedup" to the new suffix naming does
not correspond to any real split index.

Conformers matter for THIS script specifically
--------------------------------------------------
Unlike the peak-embedder-only extraction, this script feeds real 3D
coordinates (`batch[1]["atom_coords"]`) through the trunk, so which
conformer gets sampled per molecule is relevant here in a way it isn't for
`extraction/00_global_cond.py`. The clean-v2 training SLURM used
`+aug_conf=conf10` (regenerated coordinates, up to 10 conformers/molecule),
vs. an older `conf3` milestone; the v3 yaml doesn't restate an aug_conf
setting, so don't assume it's unchanged either -- match whatever `aug_conf`
the specific checkpoint you're extracting from was actually trained with
via `--aug-conf`. Left unset, this falls back to the base dataset config's
own default, which may not match a newer checkpoint.

Outputs (per split)
--------------------
    cotrain_train_layerwise.pt
    cotrain_val_layerwise.pt
    cotrain_test_layerwise.pt

Each is a dict:
    {
        "split", "ckpt", "sigma_data", "condition",
        "layers": [...], "timesteps": [...],
        "smiles": [...], "dataset": [...],
        "mol_idx": LongTensor or None,   # join key back to
                                          # cotrain_<split>_global_cond.pt
                                          # for global_cond/ECFP/properties
        "layer_timestep_data": {
            (layer, timestep): {
                "x_hidden_mean": FloatTensor [N, Dx],
                "y_hidden_mean": FloatTensor [N, Dy],
            },
            ...
        },
        "global_cond": FloatTensor [N, D] or absent (--save-global-cond),
        "ecfp": ByteTensor [N, n_bits] or absent (--save-ecfp),
        "ecfp_radius", "ecfp_nbits": present only if --save-ecfp,
        "extracted_at", "seed",
    }

Usage
-----

python extraction/02_decoder_layers.py --nmr3d-root /scratch/gpfs/ZHONGE/jc4587/research/1_chemical_landscape_analysis/8_14_26_cotrainv3_dedupOFF_s0_epoch1399/nmr-to-3d --ckpt /projects/CRYOEM/zhonglab/data_nmr/2026/ckpts/26-07-27-cotrain-v3-dedupOFF-s0/epoch1399-accuracy75.98.ckpt --save-dir /scratch/gpfs/ZHONGE/jc4587/research/cotrainv3_embeddings/26-07-27-cotrain-v3-dedupOFF-s0/epoch1399 --layers 2 5 11 --timesteps 0.001 --aug-conf conf10 --splits train val test # --n-samples-per-dataset 100

"""

from __future__ import annotations

import argparse
import contextlib
import json
import multiprocessing as mp
import random
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader as TorchDataLoader, Subset as TorchSubset
from tqdm import tqdm

# This script lives in extraction/, so sys.path[0] is that directory rather than
# the repo root and a bare `from src.common...` would not resolve. Prepending the
# root has to happen BEFORE those imports, and therefore before
# `nmr3d_import_scope` ever runs: the scope below works by evicting the cached
# `src` module so Python re-resolves it against nmr3d_root, which relies on this
# repo's `src` having been imported as a namespace package first.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.common.ckpt_meta import (bounds_overrides, c_peak_norm_overrides, read_ckpt_meta,
                                   resolve_sigma_data)
from src.common.workers import safe_n_workers


@contextlib.contextmanager
def nmr3d_import_scope(nmr3d_root: Path):
    """Makes `from src.xxx import yyy` resolve against nmr3d_root's OWN
    `src` package for the duration of the `with` block, then restores this
    repo's `src.*` modules afterward.

    Necessary because this repo's `src/` has no `__init__.py` (an implicit
    PEP 420 namespace package), while nmr-to-3d's `src/` is a regular
    package (has `__init__.py`). The `from src.common.workers import
    safe_n_workers` above always runs first (at module import time, before
    nmr3d_root is known), which permanently caches `src` in sys.modules as
    OUR namespace package. Simply having nmr3d_root on sys.path does NOT
    make that cached object resolve submodules against nmr3d_root --
    `from src.model.model import ...` then fails with `ModuleNotFoundError:
    No module named 'src.model'` even though nmr3d_root/src/model/model.py
    exists. Evicting the cached `src`/`src.*` entries forces Python to
    re-resolve `src` from scratch against the current sys.path (with
    nmr3d_root first), which finds nmr3d_root's REGULAR package instead."""
    nmr3d_root_str = str(nmr3d_root)
    if nmr3d_root_str in sys.path:
        sys.path.remove(nmr3d_root_str)
    sys.path.insert(0, nmr3d_root_str)

    saved = {k: v for k, v in sys.modules.items() if k == "src" or k.startswith("src.")}
    for k in saved:
        del sys.modules[k]
    try:
        yield
    finally:
        for k in [k for k in sys.modules if k == "src" or k.startswith("src.")]:
            del sys.modules[k]
        sys.modules.update(saved)

# -----------------------------------------------------------------------------
# Default cotrain composition for the 2026-07-20 "v3" split rebuild
# (cotrain-nmrexp-spectranp-uspto-v3.yaml).
#
# hydra_name points at the `-v3` single-source configs -- the same ones the
# cotrain bundle composes -- so the extracted corpus matches the training
# mixture exactly. Verified field-by-field against
# 26-07-27-cotrain-v3-dedupOFF-s0/epoch1399's own `dataset_args`.
#
# Padding bounds are additionally pinned to the checkpoint's MIXTURE-WIDE
# values by `bounds_overrides()`: cotrain training padded every source to the
# max across sources (200 atoms, 61/116 h/c peaks), whereas a per-source
# compose would pad spectranp to 47/98 and uspto to 28/74. This matters more
# here than for script 0 -- the trunk also sees atom padding, not just peaks.
# See src/common/ckpt_meta.py.
# -----------------------------------------------------------------------------

DEFAULT_SOURCES = [
    {
        "name": "nmrexp",
        "hydra_name": "nmrexp-v3",
        "split_suffix": "_nmrpeak_full_v3_realdedup",
    },
    {
        "name": "spectranp",
        "hydra_name": "spectranp-v3",
        "split_suffix": "_or_v3base_realdedup",
        "valid_h_key": "valid_indices_h_clean",
    },
    {
        "name": "uspto",
        "hydra_name": "uspto-v3",
        "split_suffix": "_both_full_realdedup_v3",
    },
]

SEED = 1234


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


@dataclass
class SourceSpec:
    name: str
    hydra_name: str
    split_suffix: str
    valid_h_key: Optional[str] = None
    valid_c_key: Optional[str] = None
    # Suffix to use in place of naive "<suffix>_dedup" concatenation when
    # --dedup is requested. If None and --dedup is passed for this source,
    # the run aborts with a clear error rather than guessing.
    dedup_suffix: Optional[str] = None
    extra_overrides: List[str] = field(default_factory=list)


@dataclass
class RunConfig:
    nmr3d_root: Path
    ckpt: Path
    ckpt_sigma: float
    save_dir: Path
    prefix: str
    condition: str
    sources: List[SourceSpec]
    dedup: bool
    splits: List[str]
    layers: List[int]
    timesteps: List[float]
    aug_conf: Optional[str]
    batch_size: Optional[int]
    ecfp_radius: int
    ecfp_nbits: int
    save_per_source: bool
    save_global_cond: bool
    save_ecfp: bool
    n_ecfp_workers: int
    n_samples_per_dataset: Optional[int]
    sample_strategy: str
    # The c_peak_norm_args hydra overrides actually applied, read off the
    # checkpoint. Recorded in the manifest so a run states which 13C
    # normalization its embeddings were extracted under.
    c_peak_norm_overrides: List[str]
    # The dataset_args padding-bound overrides actually applied, read off the
    # checkpoint. Training padded every source to the mixture-wide max, so
    # these are what make a per-source re-extraction match it.
    bounds_overrides: List[str]
    device: str
    seed: int


def parse_args(argv: Optional[Sequence[str]] = None) -> RunConfig:
    p = argparse.ArgumentParser(
        description="Extract multi-layer, multi-timestep decoder hidden "
                    "states (+ optional global_cond/ECFP) for the cotrain "
                    "dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--nmr3d-root", type=Path, required=True)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--ckpt-sigma", type=float, default=None,
                   help="Optional. sigma_data is read from the checkpoint's "
                        "own hyper_parameters; pass this only to assert a "
                        "value, and it will warn -- and defer to the "
                        "checkpoint -- on a mismatch. See the module "
                        "docstring for why it cannot corrupt the embeddings.")
    p.add_argument("--save-dir", type=Path, required=True)
    p.add_argument("--prefix", type=str, default="cotrain")
    p.add_argument("--condition", type=str, default="hcpeak")
    p.add_argument("--sources-json", type=Path, default=None,
                   help="Optional JSON overriding the default 3-source "
                        "composition (list of {name, hydra_name, "
                        "split_suffix, valid_h_key, valid_c_key, "
                        "dedup_suffix, extra_overrides} dicts).")
    p.add_argument("--dedup", action="store_true",
                   help="Use each source's configured `dedup_suffix` "
                        "instead of its training `split_suffix`. Aborts if "
                        "a requested source has no `dedup_suffix` "
                        "configured.")
    p.add_argument("--no-ckpt-bounds", action="store_true",
                   help="Do NOT pin the padding/tensor-shape bounds "
                        "(max_n_atoms, h/c_max_n_peaks, h_max_nH) to the "
                        "checkpoint's mixture-wide values, using each "
                        "source's own hydra config bounds instead. Training "
                        "padded every source to the MAX across sources, so "
                        "the default reproduces that; per-source bounds pad "
                        "spectranp/uspto shorter than the model ever saw. "
                        "Escape hatch only.")
    p.add_argument("--no-ckpt-cpeak-norm", action="store_true",
                   help="Do NOT pin dataset_args.c_peak_norm_args (13C "
                        "symmetry-merge) to the checkpoint's own "
                        "training-time values, leaving the hydra config "
                        "defaults in place. Because this normalization is "
                        "applied at val/test too, that means extracting from "
                        "spectra the model never saw unless the config "
                        "default already matches. Escape hatch only.")
    p.add_argument("--apply-cpeak-norm-overrides", action="store_true",
                   help="DEPRECATED. c_peak_norm_args is now read from the "
                        "checkpoint, so this flag is unnecessary; it is "
                        "accepted only when it agrees with the checkpoint "
                        "(dedup_p=1.0) and errors otherwise.")
    p.add_argument("--aug-conf", type=str, default=None,
                   help="+aug_conf Hydra override (e.g. 'conf10'), applied "
                        "to every source. Match whatever the checkpoint "
                        "you're extracting from was actually trained with "
                        "(the clean-v2 SLURM used conf10; v3's aug_conf "
                        "setting is not documented in its yaml -- verify "
                        "separately). Left unset, falls back to the base "
                        "dataset config's own default. See module docstring.")
    p.add_argument("--splits", nargs="+", default=["train"],
                   choices=["train", "val", "test"])

    p.add_argument("--layers", nargs="+", type=int,
                   default=[-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                   help="Trunk layers to extract hidden states from (-1 == "
                        "final layer).")
    p.add_argument("--timesteps", nargs="+", type=float, default=[0.001, 0.5, 1.0],
                   help="Diffusion timesteps (in [0,1]) to noise coordinates "
                        "to before the trunk forward pass. t<=0.001 uses "
                        "clean coordinates directly.")

    p.add_argument("--save-global-cond", action="store_true",
                   help="Also save the peak embedder's pooled global_cond "
                        "(duplicates what extraction/00_global_cond.py already "
                        "saves once per molecule; off by default -- join on "
                        "mol_idx instead).")
    p.add_argument("--save-ecfp", action="store_true",
                   help="Also compute+save ECFPs (likewise already saved by "
                        "extraction/00_global_cond.py; off by default).")
    p.add_argument("--ecfp-radius", type=int, default=2)
    p.add_argument("--ecfp-nbits", type=int, default=2048)
    p.add_argument("--save-per-source", action="store_true")
    p.add_argument("--n-ecfp-workers", type=int, default=safe_n_workers())
    p.add_argument("--n-samples-per-dataset", type=int, default=5000,
                   help="Cap on molecules processed per source dataset per "
                        "split -- this extraction runs len(layers)*"
                        "len(timesteps) trunk forward passes per batch, so "
                        "full-scale runs are typically infeasible. Pass 0 "
                        "to disable the cap.")
    p.add_argument("--sample-strategy", type=str, default="random", choices=["random", "prefix"],
                   help="How --n-samples-per-dataset picks its molecules. "
                        "'random' (default) draws a uniform random subset of "
                        "split indices, seeded by --seed, so val/test are real "
                        "samples rather than an index-ordered prefix. 'prefix' "
                        "restores the old behaviour of consuming the loader "
                        "until the cap is hit.")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Override the dataloader batch size for all splits "
                        "(applied to every source via "
                        "dataset_args.batch_size for train/val and "
                        "dataset_args.test_args.test_batch_size for test). "
                        "Left unset, falls back to the base dataset "
                        "config's own default (128 for the cotrain sources).")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args(argv)

    sources_raw = json.loads(Path(args.sources_json).read_text()) if args.sources_json else DEFAULT_SOURCES
    sources = [SourceSpec(**s) for s in sources_raw]

    if args.dedup:
        new_sources = []
        for s in sources:
            if s.dedup_suffix is None:
                raise ValueError(
                    f"--dedup was requested but source '{s.name}' has no "
                    f"`dedup_suffix` configured. Set the correct "
                    f"zero-shot-eval suffix explicitly via --sources-json."
                )
            new_sources.append(SourceSpec(
                name=s.name, hydra_name=s.hydra_name, split_suffix=s.dedup_suffix,
                valid_h_key=s.valid_h_key, valid_c_key=s.valid_c_key,
                dedup_suffix=s.dedup_suffix, extra_overrides=s.extra_overrides,
            ))
        sources = new_sources

    # sigma_data + the 13C symmetry-merge settings both come from the
    # checkpoint's own hyper_parameters, not from constants pasted into a
    # SLURM. `c_peak_norm_args` is a single GLOBAL dataset_args block (not
    # per-source), so it is pinned on every source identically.
    ckpt_meta = read_ckpt_meta(args.ckpt)
    print(ckpt_meta.summary())
    resolved_sigma = resolve_sigma_data(ckpt_meta, args.ckpt_sigma)

    cpeak_overrides = [] if args.no_ckpt_cpeak_norm else c_peak_norm_overrides(ckpt_meta)
    bound_overrides = [] if args.no_ckpt_bounds else bounds_overrides(ckpt_meta)
    if args.apply_cpeak_norm_overrides:
        ckpt_dedup_p = ckpt_meta.c_peak_norm.get("dedup_p")
        if ckpt_dedup_p is not None and float(ckpt_dedup_p) != 1.0:
            raise ValueError(
                f"--apply-cpeak-norm-overrides forces the 13C symmetry-merge ON, but "
                f"{args.ckpt.name} was trained with dedup_p={ckpt_dedup_p} (merge OFF). "
                f"Extracting with it on would feed the model spectra it never saw. This "
                f"flag is deprecated -- c_peak_norm_args is now read from the checkpoint, "
                f"so just drop it.")
        print("[ckpt-meta] --apply-cpeak-norm-overrides is deprecated and redundant; "
              "c_peak_norm_args is read from the checkpoint.")
    for s in sources:
        s.extra_overrides = list(s.extra_overrides) + cpeak_overrides + bound_overrides
    print(f"[ckpt-meta] c_peak_norm overrides applied to every source: "
          f"{cpeak_overrides or '(none -- hydra config defaults)'}")
    print(f"[ckpt-meta] padding bounds pinned on every source: "
          f"{bound_overrides or '(none -- per-source hydra config bounds)'}")

    if args.aug_conf:
        for s in sources:
            s.extra_overrides = list(s.extra_overrides) + [f"+aug_conf={args.aug_conf}"]

    if args.batch_size is not None:
        for s in sources:
            s.extra_overrides = list(s.extra_overrides) + [
                f"dataset_args.batch_size={args.batch_size}",
                f"dataset_args.test_args.test_batch_size={args.batch_size}",
            ]

    n_cap = None if (args.n_samples_per_dataset is None or args.n_samples_per_dataset <= 0) else args.n_samples_per_dataset

    return RunConfig(
        nmr3d_root=args.nmr3d_root, ckpt=args.ckpt, ckpt_sigma=resolved_sigma,
        save_dir=args.save_dir, prefix=args.prefix, condition=args.condition,
        sources=sources, dedup=args.dedup, splits=args.splits,
        layers=args.layers, timesteps=args.timesteps, aug_conf=args.aug_conf,
        batch_size=args.batch_size,
        ecfp_radius=args.ecfp_radius, ecfp_nbits=args.ecfp_nbits,
        save_per_source=args.save_per_source,
        save_global_cond=args.save_global_cond, save_ecfp=args.save_ecfp,
        n_ecfp_workers=args.n_ecfp_workers, n_samples_per_dataset=n_cap,
        sample_strategy=args.sample_strategy,
        c_peak_norm_overrides=cpeak_overrides,
        bounds_overrides=bound_overrides,
        device=args.device, seed=args.seed,
    )


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------


def load_model(ckpt: Path, device: str, nmr3d_root: Path):
    """Returns (model, trunk, peak_embedder, diffusion_module).

    `trunk` is the score-model transformer (callable with
    `extract_features=True, extract_layer=...`); `diffusion_module` is
    whatever object exposes `.interpolate` / `._apply_coordinate_transform`
    (the flow-matching noising utilities) -- both are resolved with a
    fallback in case the Lightning module wrapping differs across
    checkpoint versions.
    """
    with nmr3d_import_scope(nmr3d_root):
        from src.model.model import NMRTo3DStructureElucidation

        model = NMRTo3DStructureElucidation.load_from_checkpoint(str(ckpt), map_location=device)
        model.eval()
        model.to(device)

        if hasattr(model, "model") and hasattr(model.model, "score_model"):
            trunk = model.model.score_model
        else:
            trunk = model.score_model

        if hasattr(model, "model") and hasattr(model.model, "interpolate"):
            diffusion_module = model.model
        else:
            diffusion_module = model

        peak_embedder = trunk.y_embedder
    return model, trunk, peak_embedder, diffusion_module


# -----------------------------------------------------------------------------
# Datamodule construction
# -----------------------------------------------------------------------------


def build_datamodule(cfg_dir: str, source: SourceSpec, sigma_data: float, condition: str,
                      nmr3d_root: Path):
    with nmr3d_import_scope(nmr3d_root):
        from hydra import compose, initialize_config_dir
        from src.data.datamodule import NMRDataModule

        overrides = [
            f"+data={source.hydra_name}",
            f"+condition={condition}",
            f"dataset_args.split_indices_suffix={source.split_suffix}",
            f"dataset_args.sigma_data={sigma_data}",
        ]
        if source.valid_h_key is not None:
            overrides.append(f"dataset_args.valid_h_key={source.valid_h_key}")
        if source.valid_c_key is not None:
            overrides.append(f"dataset_args.valid_c_key={source.valid_c_key}")
        overrides.extend(source.extra_overrides)

        try:
            with initialize_config_dir(cfg_dir, version_base=None):
                cfg = compose(config_name="config", overrides=overrides)
        except Exception as e:
            raise RuntimeError(
                f"Failed to compose Hydra config for source '{source.name}' "
                f"(hydra_name='{source.hydra_name}') with overrides={overrides}.\n"
                f"The default sources' hydra_name/valid_h_key were confirmed "
                f"working for the clean-v2 composition, and the v3 suffixes "
                f"were only updated by reference to the v3 yaml (not yet "
                f"smoke-tested through this script) -- if you're using "
                f"DEFAULT_SOURCES unmodified, check --apply-cpeak-norm-overrides "
                f"and --aug-conf first (both are the least-certain override "
                f"paths) before suspecting the base source config.\n"
                f"Original error: {e}"
            ) from e

        dm = NMRDataModule(cfg.dataset_args)
    return dm


def get_split_dataloader(dm, split: str):
    if split in ("train", "val"):
        dm.prepare_data()
        dm.setup("fit")
        return dm.train_dataloader() if split == "train" else dm.val_dataloader()
    elif split == "test":
        dm.prepare_data()
        dm.setup("test")
        return dm.test_dataloader()
    raise ValueError(f"Unknown split: {split}")


def subsample_dataloader(dataloader, n: Optional[int], seed: int):
    """Uniform random subset of `n` molecules, drawn at the INDEX level.

    The alternative -- iterating the loader and breaking once n molecules have
    been seen -- takes the first n in loader order, which is only a real sample
    where the loader shuffles. `train_dataloader` does (dataset_args.shuffle),
    but `val_dataloader`/`test_dataloader` hardcode shuffle=False, so a prefix
    there is an index-ordered slice: whatever molecules happen to sit at the
    front of the split. Sampling the indices makes all three splits behave the
    same and reproducibly, keyed on --seed rather than on torch's global RNG
    state at iteration time."""
    dataset = dataloader.dataset
    total = len(dataset)
    if n is None or n >= total:
        return dataloader, total
    idx = np.random.default_rng(seed).choice(total, size=n, replace=False)
    subset = TorchSubset(dataset, sorted(int(i) for i in idx))
    return (
        TorchDataLoader(
            subset,
            batch_size=dataloader.batch_size,
            num_workers=dataloader.num_workers,
            pin_memory=dataloader.pin_memory,
            collate_fn=dataloader.collate_fn,
            shuffle=False,
        ),
        total,
    )



def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask[..., None].float()
    return (x * mask).sum(1) / mask.sum(1).clamp(min=1)


# -----------------------------------------------------------------------------
# ECFP computation (only used if --save-ecfp)
# -----------------------------------------------------------------------------


def _ecfp_worker(args: Tuple[str, int, int]) -> np.ndarray:
    smiles, radius, n_bits = args
    from rdkit import Chem
    from rdkit.Chem import AllChem

    bits = np.zeros(n_bits, dtype=np.uint8)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"[ecfp] WARNING: could not parse SMILES, using zero fingerprint: {smiles!r}", file=sys.stderr)
        return bits
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    bits[list(fp.GetOnBits())] = 1
    return bits


def compute_ecfps(smiles_list: List[str], radius: int, n_bits: int, n_workers: int) -> torch.Tensor:
    tasks = [(s, radius, n_bits) for s in smiles_list]
    if n_workers <= 1:
        results = [_ecfp_worker(t) for t in tqdm(tasks, desc="ECFP (serial)")]
    else:
        with mp.Pool(n_workers) as pool:
            results = list(tqdm(pool.imap(_ecfp_worker, tasks, chunksize=256),
                                total=len(tasks), desc=f"ECFP ({n_workers} workers)"))
    return torch.from_numpy(np.stack(results, axis=0))


# -----------------------------------------------------------------------------
# Per-source, per-split multi-layer extraction
# -----------------------------------------------------------------------------


def extract_layerwise_for_split(model, trunk, peak_embedder, diffusion_module,
                                 dm, split: str, layers: List[int], timesteps: List[float],
                                 device: str, n_samples_cap: Optional[int],
                                 save_global_cond: bool, sample_strategy: str = "random",
                                 seed: int = SEED) -> Dict:
    dataloader = get_split_dataloader(dm, split)
    if sample_strategy == "random":
        dataloader, n_total = subsample_dataloader(dataloader, n_samples_cap, seed)
        if n_samples_cap is not None and n_samples_cap < n_total:
            print(f"    sampling {n_samples_cap} of {n_total} molecules at random (seed={seed})")
        n_samples_cap = None  # the cap is already applied at the index level

    smiles_out: List[str] = []
    mol_idx_out: List[int] = []
    global_cond_out: List[torch.Tensor] = []
    have_mol_idx = True

    layer_timestep_out = {
        (layer, t): {"x_hidden_mean": [], "y_hidden_mean": []}
        for layer in layers for t in timesteps
    }

    n_seen = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"  [{split}] forward pass (multi-layer)"):
            if n_samples_cap is not None and n_seen >= n_samples_cap:
                break

            model_inputs, smiles = batch[0]
            coords = batch[1]["atom_coords"].to(device)

            atom_mask = model_inputs["atom_mask"].to(device)
            atom_one_hot = model_inputs["atom_one_hot"].float().to(device)
            condition = {k: v.to(device) for k, v in model_inputs["condition"].items()}

            B = coords.shape[0]

            if save_global_cond:
                global_cond, _, _ = peak_embedder(condition)
                global_cond_out.append(global_cond.cpu())

            for layer in layers:
                for t in timesteps:
                    times = torch.full((B,), t, device=device)

                    # Match the diffusion training/sampling noising process.
                    if t <= 0.001:
                        coords_in = coords
                    else:
                        coords_in = diffusion_module.interpolate(
                            coords, torch.randn_like(coords), times,
                        )
                        coords_in = diffusion_module._apply_coordinate_transform(
                            coords_in, atom_mask, "centering",
                        )

                    feat = trunk(
                        r_noisy=coords_in,
                        times=times,
                        model_inputs={
                            "atom_mask": atom_mask,
                            "atom_one_hot": atom_one_hot,
                            "condition": condition,
                        },
                        extract_features=True,
                        extract_layer=layer,
                    )

                    layer_timestep_out[(layer, t)]["x_hidden_mean"].append(
                        masked_mean(feat["x_hidden"], atom_mask).cpu()
                    )
                    layer_timestep_out[(layer, t)]["y_hidden_mean"].append(
                        masked_mean(feat["y_hidden"], feat["y_mask"]).cpu()
                    )

            smiles_out.extend(smiles)

            if have_mol_idx:
                mol_idx_batch = model_inputs.get("mol_idx") if isinstance(model_inputs, dict) else None
                if mol_idx_batch is not None:
                    mol_idx_out.extend(mol_idx_batch.detach().cpu().tolist())
                else:
                    have_mol_idx = False
                    mol_idx_out = []

            n_seen += B
            print(f"    {min(n_seen, n_samples_cap) if n_samples_cap else n_seen} molecules processed", end="\r")

    print()  # newline after the \r progress counter

    n_final = len(smiles_out)
    result = {
        "smiles": smiles_out,
        "mol_idx": (
            torch.tensor(mol_idx_out, dtype=torch.long)
            if have_mol_idx and len(mol_idx_out) == n_final else None
        ),
        "layer_timestep_data": {
            key: {name: torch.cat(vals, dim=0)[:n_final] for name, vals in val.items()}
            for key, val in layer_timestep_out.items()
        },
    }
    if save_global_cond:
        result["global_cond"] = torch.cat(global_cond_out, dim=0)[:n_final]
    return result


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> None:
    cfg = parse_args(argv)
    set_seed(cfg.seed)

    config_dir = str(cfg.nmr3d_root / "configs")
    cfg.save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("Cotrain layerwise (decoder) embedding extraction "
          "(2026-07-20 cotrain-v3 composition)")
    print(f"  ckpt        : {cfg.ckpt}")
    print(f"  ckpt sigma  : {cfg.ckpt_sigma}  (read from the checkpoint)")
    print(f"  condition   : {cfg.condition}")
    print(f"  sources     : {[(s.name, s.hydra_name, s.split_suffix) for s in cfg.sources]}")
    print(f"  splits      : {cfg.splits}")
    print(f"  layers      : {cfg.layers}")
    print(f"  timesteps   : {cfg.timesteps}")
    print(f"  aug_conf    : {cfg.aug_conf or '(dataset default)'}")
    print(f"  batch_size  : {cfg.batch_size or '(dataset default, 128)'}")
    print(f"  n_samples/ds: {cfg.n_samples_per_dataset or 'ALL (expensive!)'}")
    print(f"  sampling    : {cfg.sample_strategy}")
    print(f"  save global_cond / ecfp: {cfg.save_global_cond} / {cfg.save_ecfp}")
    print(f"  c_peak_norm : {cfg.c_peak_norm_overrides or '(hydra config defaults)'}")
    print(f"  bounds      : {cfg.bounds_overrides or '(per-source hydra config bounds)'}")
    if cfg.aug_conf is None:
        print("  [reminder] --aug-conf is unset. If this checkpoint was "
              "trained with a non-default conformer augmentation (e.g. "
              "clean-v2's conf10), pass --aug-conf to match it -- v3's "
              "aug_conf setting isn't documented in its yaml, so verify "
              "separately for a v3 checkpoint.")
    print("=" * 78)

    model, trunk, peak_embedder, diffusion_module = load_model(cfg.ckpt, cfg.device, cfg.nmr3d_root)

    combined: Dict[str, Dict] = {
        split: {
            "smiles": [], "dataset": [], "mol_idx": [], "global_cond": [],
            "layer_timestep_data": {(l, t): {"x_hidden_mean": [], "y_hidden_mean": []}
                                     for l in cfg.layers for t in cfg.timesteps},
        }
        for split in cfg.splits
    }
    combined_has_mol_idx = {split: True for split in cfg.splits}

    for source in cfg.sources:
        print(f"\n### Source: {source.name} ({source.hydra_name}, "
              f"suffix={source.split_suffix}, valid_h_key={source.valid_h_key}) ###")
        dm = build_datamodule(config_dir, source, cfg.ckpt_sigma, cfg.condition, cfg.nmr3d_root)

        for split in cfg.splits:
            out = extract_layerwise_for_split(
                model, trunk, peak_embedder, diffusion_module, dm, split,
                cfg.layers, cfg.timesteps, cfg.device, cfg.n_samples_per_dataset,
                cfg.save_global_cond, sample_strategy=cfg.sample_strategy, seed=cfg.seed,
            )
            n = len(out["smiles"])
            print(f"  [{source.name}/{split}] {n} molecules")

            ecfp = (
                compute_ecfps(out["smiles"], cfg.ecfp_radius, cfg.ecfp_nbits, cfg.n_ecfp_workers)
                if cfg.save_ecfp else None
            )

            if cfg.save_per_source:
                per_source_path = cfg.save_dir / f"{source.name}_{split}_layerwise.pt"
                per_source_dict = {
                    "split": split, "dataset": source.name, "ckpt": str(cfg.ckpt),
                    "sigma_data": cfg.ckpt_sigma, "condition": cfg.condition,
                    "layers": cfg.layers, "timesteps": cfg.timesteps,
                    "split_suffix": source.split_suffix,
                    "smiles": out["smiles"], "mol_idx": out["mol_idx"],
                    "layer_timestep_data": out["layer_timestep_data"],
                    "extracted_at": datetime.now(timezone.utc).isoformat(), "seed": cfg.seed,
                }
                if cfg.save_global_cond:
                    per_source_dict["global_cond"] = out["global_cond"]
                if cfg.save_ecfp:
                    per_source_dict.update({
                        "ecfp": ecfp, "ecfp_radius": cfg.ecfp_radius, "ecfp_nbits": cfg.ecfp_nbits,
                    })
                torch.save(per_source_dict, per_source_path)
                print(f"    -> saved {per_source_path}")

            combined[split]["smiles"].extend(out["smiles"])
            combined[split]["dataset"].extend([source.name] * n)
            if cfg.save_global_cond:
                combined[split]["global_cond"].append(out["global_cond"])
            for key, val in out["layer_timestep_data"].items():
                combined[split]["layer_timestep_data"][key]["x_hidden_mean"].append(val["x_hidden_mean"])
                combined[split]["layer_timestep_data"][key]["y_hidden_mean"].append(val["y_hidden_mean"])
            if out["mol_idx"] is not None:
                combined[split]["mol_idx"].append(out["mol_idx"])
            else:
                combined_has_mol_idx[split] = False

    manifest = {
        "ckpt": str(cfg.ckpt), "ckpt_sigma": cfg.ckpt_sigma, "condition": cfg.condition,
        "sources": [asdict(s) for s in cfg.sources], "dedup": cfg.dedup,
        "layers": cfg.layers, "timesteps": cfg.timesteps, "aug_conf": cfg.aug_conf,
        "batch_size": cfg.batch_size,
        "c_peak_norm_overrides": list(cfg.c_peak_norm_overrides),
        "bounds_overrides": list(cfg.bounds_overrides),
        "save_global_cond": cfg.save_global_cond, "save_ecfp": cfg.save_ecfp,
        "n_samples_per_dataset": cfg.n_samples_per_dataset,
        "sample_strategy": cfg.sample_strategy, "seed": cfg.seed,
        "extracted_at": datetime.now(timezone.utc).isoformat(), "outputs": [],
    }

    for split in cfg.splits:
        acc = combined[split]
        n_total = len(acc["smiles"])
        if n_total == 0:
            print(f"[warn] split '{split}' has 0 molecules across all sources, skipping save.")
            continue

        mol_idx_all = (
            torch.cat(acc["mol_idx"], dim=0)
            if combined_has_mol_idx[split] and len(acc["mol_idx"]) == len(cfg.sources) else None
        )

        layer_timestep_all = {
            key: {name: torch.cat(vals, dim=0) for name, vals in val.items()}
            for key, val in acc["layer_timestep_data"].items()
        }

        out_dict = {
            "split": split, "ckpt": str(cfg.ckpt), "sigma_data": cfg.ckpt_sigma,
            "condition": cfg.condition, "layers": cfg.layers, "timesteps": cfg.timesteps,
            "sources": [asdict(s) for s in cfg.sources],
            "smiles": acc["smiles"], "dataset": acc["dataset"], "mol_idx": mol_idx_all,
            "layer_timestep_data": layer_timestep_all,
            "extracted_at": datetime.now(timezone.utc).isoformat(), "seed": cfg.seed,
        }
        if cfg.save_global_cond and acc["global_cond"]:
            out_dict["global_cond"] = torch.cat(acc["global_cond"], dim=0)
        if cfg.save_ecfp:
            ecfp_all = compute_ecfps(acc["smiles"], cfg.ecfp_radius, cfg.ecfp_nbits, cfg.n_ecfp_workers)
            out_dict.update({
                "ecfp": ecfp_all, "ecfp_radius": cfg.ecfp_radius, "ecfp_nbits": cfg.ecfp_nbits,
            })

        out_path = cfg.save_dir / f"{cfg.prefix}_{split}_layerwise.pt"
        torch.save(out_dict, out_path)
        manifest["outputs"].append({"split": split, "path": str(out_path), "n_molecules": n_total})
        print(f"\n✓ Saved {out_path} ({n_total} molecules, "
              f"{len(layer_timestep_all)} (layer,timestep) combos)")

    manifest_path = cfg.save_dir / f"{cfg.prefix}_layerwise_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\n✓ Wrote manifest: {manifest_path}")
    print("Done.")


if __name__ == "__main__":
    main()