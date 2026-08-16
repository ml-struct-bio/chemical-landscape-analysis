#!/usr/bin/env python
"""
00_global_cond.py
==============================

End-to-end, checkpoint-agnostic extraction of:
    - SMILES
    - NMR peak-embedder ("global_cond") embeddings
    - ECFP (Morgan) fingerprints
    - source-dataset labels

for every molecule in the *cotrain* dataset, for the train / val / test
splits.

Version lineage: original -> clean/clean-v2 (2026-07-17) -> v3 (2026-07-20)
------------------------------------------------------------------------------
This has now been updated twice. Original composition:
`cotrain-nmrexp-nmrpeak-spectranp-uspto.yaml`. The 2026-07-17 rebuild
("clean"/"clean-v2") introduced the finalized, leakage-free, triple-cleaned
mixture described below; this version updates it again for the 2026-07-20
"v3" split rebuild (`cotrain-nmrexp-spectranp-uspto-v3.yaml`), which keeps
the same overall recipe but rebuilds the splits themselves so the
head-to-head NMRPeak-reproduce test set is mutually novel, restores
spectranp's original `_or` split (folding a data-recovery batch into TRAIN
only), and re-deduplicates all three VAL sets against NMRPeak's released
test + real-* benchmarks. Per-source datadir paths are UNCHANGED across all
three versions (only split_indices_suffix / valid_h_key move), so the
the composition below now uses the `-v3` configs directly (see below).

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

CONFIRMED for clean-v2 via a --limit 100 test extraction against
cotrain-clean-v2-epoch0349-accuracy68_06.ckpt (train counts matched the
clean-v2 training SLURM's stated numbers exactly for all three sources;
also confirmed `dataset_args.valid_h_key=<key>` is a valid override on a
standalone dataset config, and is actually applied, not silently ignored).
NOT YET independently confirmed for v3's specific suffix strings --
recommend the same kind of --n-samples-per-dataset (or --limit) smoke test
before a full run, checking the printed per-source train/val/test counts
against the v3 yaml's own stated numbers (see the comment block above
DEFAULT_SOURCES).

RESOLVED (2026-08-11): whether `dataset_args.c_peak_norm_args.*` (the 13C
symmetry-merge) applies to v3 is no longer a guess. It is read straight off
the checkpoint's own `hyper_parameters` -- see the c_peak_norm_args section
below and `src/common/ckpt_meta.py`.

Why the valid_h_key override matters (per your colleague's note): the
*default* validity index for a dataset is `valid_indices_h` (analogously
`valid_indices_c`). Both the clean-v2 and v3 yamls point spectranp at
`valid_indices_h_clean` instead, because 2 corrupt-1H molecules with
nH=189/177 would otherwise sneak into `valid_indices_h` un-gated. Skipping
this override doesn't crash -- it silently uses the wrong (dirtier)
molecule set. This script applies `valid_h_key`/`valid_c_key` per-source
when configured, falling back to each dataset's own default validity key
when not set.

sigma_data -- read from the checkpoint, and NOT what scales the model
------------------------------------------------------------------------
`sigma_data: null` in both the clean-v2 and v3 yamls means it's a *computed*
weighted-RMS mixture over the sources -- baked into whatever checkpoint gets
trained on THAT mixture (2.8418 for every cotrain-v3 ckpt).

An earlier revision of this docstring claimed a stale `--ckpt-sigma` would
"silently produce wrong embeddings". That is NOT true, and the correction
matters because it's why a wrong value can sit here unnoticed:

  * The value the model uses is `cfg.diffusion_process_args.sigma_data`,
    which `load_from_checkpoint` restores from the ckpt's own saved
    hyper_parameters. `nmr-to-3d/src/main.py` copies datamodule -> diffusion
    only at TRAINING construction, and only when the diffusion value is None.
  * What we pass here lands on `dataset_args.sigma_data`, i.e. on
    `NMRDataModule.sigma_data`, which is read nowhere during a forward pass
    and never reaches a batch tensor.

So the override's only live effect is to skip the datamodule's expensive
weighted-RMS recompute for `sigma_data: null` configs. Its value affects the
MANIFEST, not the embeddings -- which is exactly why it must still be right.
`--ckpt-sigma` is therefore optional now: the value is read from the
checkpoint, and passing one only asserts it (mismatch warns and defers to the
checkpoint). See `src/common/ckpt_meta.py`.

The setting that DOES change the extracted tensors is `c_peak_norm_args`,
below.

`--dedup` behavior changed
----------------------------
The old `--dedup` flag naively appended `"_dedup"` to whatever suffix was
configured (correct for the old zero-shot-eval suffixes like `_or_dedup`).
Blindly appending `"_dedup"` to e.g. `_nmrpeak_full_realdedup` does NOT
correspond to any real split index and will raise the same class of
`KeyError` seen with `_or_recon_heavy` previously. `--dedup` now requires
an explicit `dedup_suffix` configured per source (via `--sources-json`);
if none is set for a requested source, the script raises a clear error
instead of silently building a broken suffix.

c_peak_norm_args (13C symmetry-merge) -- now READ FROM THE CHECKPOINT
-----------------------------------------------------------------------------------
This is the setting that genuinely changes what gets extracted. Per
`nmr-to-3d/configs/config.yaml` it is deliberately a NORMALIZATION rather
than an augmentation: `augment_c_peaks` applies it even when `augment=False`,
so it reaches val/test/sampling too. Extract with a `dedup_p` that differs
from training and the model sees spectra it never saw -- silently, with no
error.

There is no correct default across checkpoint vintages (the 26-05-01
baselines predate the block entirely; clean-v2 turned the merge on for
spectranp; v3 ships both variants), so this script no longer guesses. It
reads `dataset_args.c_peak_norm_args` off the checkpoint's own
hyper_parameters and pins every source to it. `c_peak_norm_args` is a single
GLOBAL `dataset_args` block, not a per-source one, so it is applied
uniformly.

That also settles what the cotrain-v3 `dedupOFF`/`dedupON` checkpoint
families differ in -- exactly this:

    26-07-27-cotrain-v3-dedupOFF-s0    sigma 2.8418   dedup_p 0.0
    26-07-27-cotrain-v3-dedupON-s0     sigma 2.8418   dedup_p 1.0
    26-07-20-cotrain-v3-dedupOFF-s42   sigma 2.8418   dedup_p 0.0

`--apply-cpeak-norm-overrides` is deprecated (it now errors if it contradicts
the checkpoint). `--no-ckpt-cpeak-norm` is the escape hatch that falls back
to the hydra config defaults.

Related, but out of scope for THIS script: training SLURMs for these
compositions may also set `+aug_conf` (e.g. clean-v2 used `conf10`) and
model hyperparameters. The model hyperparameters are restored automatically
from the checkpoint by `load_from_checkpoint` (this script never
reconstructs the model via Hydra), so they need no action here. `aug_conf`
only affects which conformer/3D-coordinate pool gets sampled -- irrelevant
to this script (peak-embedder output never touches 3D coordinates), but
relevant to `extraction/02_decoder_layers.py`, which does noise real coordinates through
the trunk; match whatever `aug_conf` the checkpoint was trained with there.

Outputs
-------
    cotrain_train_global_cond.pt
    cotrain_val_global_cond.pt
    cotrain_test_global_cond.pt

Each is a dict:
    {
        "split": "train" | "val" | "test",
        "ckpt": <path>,
        "sigma_data": <float>,
        "condition": <str, e.g. "hcpeak">,
        "sources": [{"name", "hydra_name", "split_suffix", "valid_h_key",
                     "valid_c_key", "extra_overrides"}, ...],
        "smiles": List[str],
        "dataset": List[str]        # source name per molecule
        "mol_idx": LongTensor or None,
        "global_cond": FloatTensor [N, D],
        "ecfp": ByteTensor [N, n_bits]   # 0/1 per bit
        "ecfp_radius": int,
        "ecfp_nbits": int,
        "layer_reps": FloatTensor [N, ...] or absent (--save-layer-reps),
        "extracted_at": ISO timestamp,
        "seed": int,
    }

A companion `<prefix>_manifest.json` is also written with the full run
config for provenance.

Usage
-----
python extraction/00_global_cond.py --nmr3d-root /scratch/gpfs/ZHONGE/jc4587/research/1_chemical_landscape_analysis/8_14_26_cotrainv3_dedupOFF_s0_epoch1399/nmr-to-3d --ckpt /projects/CRYOEM/zhonglab/data_nmr/2026/ckpts/26-07-27-cotrain-v3-dedupOFF-s0/epoch1399-accuracy75.98.ckpt --save-dir /scratch/gpfs/ZHONGE/jc4587/research/cotrainv3_embeddings/26-07-27-cotrain-v3-dedupOFF-s0/epoch1399 --prefix cotrain
"""

from __future__ import annotations

import argparse
import contextlib
import json
import multiprocessing as mp
import random
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
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
# hydra_name points at the `-v3` single-source configs, NOT the base
# `nmrexp` / `spectranp-760k` / `uspto` ones an earlier revision used. The
# `-v3` yamls are the same configs the cotrain bundle composes, so the
# extracted corpus matches the training mixture exactly rather than
# approximately. Verified field-by-field against
# 26-07-27-cotrain-v3-dedupOFF-s0/epoch1399's own `dataset_args`: datadir,
# split_indices_suffix, atom_decoder, remove_h/remove_stereo/known_atoms,
# valid_h_key, c_peak_norm_args and all four padding bounds agree. (The
# ckpt records datadir as `/data/zx8205/...`, the training node's mount of
# the same `/projects/CRYOEM/...` datasets these yamls name.)
#
# Padding bounds are additionally pinned to the checkpoint's MIXTURE-WIDE
# values by `bounds_overrides()` -- cotrain training padded every source to
# the max across sources (200 atoms, 61/116 h/c peaks), whereas a per-source
# compose would pad spectranp to 47/98 and uspto to 28/74. See
# src/common/ckpt_meta.py.
#
# v3 expected train counts (from the v3 yaml's own comments), to sanity
# check against a smoke-test's printed "### Source ...: train=..." line:
#   nmrexp:    train=1,187,838 (val=56,082,  test=123,801)
#   spectranp: train=674,761   (val=34,226,  test=35,129)
#   uspto:     train=662,342   (val=34,730,  test=78,645)
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
    # Escape hatch for any other per-source Hydra override this config
    # revision (or a future one) needs, e.g. "dataset_args.max_n_atoms=200".
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
    ecfp_radius: int
    ecfp_nbits: int
    save_layer_reps: bool
    save_per_source: bool
    n_ecfp_workers: int
    limit: Optional[int]
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
        description="Extract SMILES / global_cond embeddings / ECFPs for the cotrain dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--nmr3d-root", type=Path, required=True,
                   help="Path to the nmr-to-3d repo root.")
    p.add_argument("--ckpt", type=Path, required=True,
                   help="Path to the Lightning checkpoint to load.")
    p.add_argument("--ckpt-sigma", type=float, default=None,
                   help="Optional. sigma_data is read from the checkpoint's "
                        "own hyper_parameters "
                        "(diffusion_process_args.sigma_data); pass this only "
                        "to assert a value, and it will warn -- and defer to "
                        "the checkpoint -- on a mismatch. See the module "
                        "docstring for why this value cannot corrupt the "
                        "embeddings, only the recorded provenance.")
    p.add_argument("--save-dir", type=Path, required=True,
                   help="Directory to write output .pt files + manifest.")
    p.add_argument("--prefix", type=str, default="cotrain",
                   help="Filename prefix, e.g. '<prefix>_train_global_cond.pt'.")
    p.add_argument("--condition", type=str, default="hcpeak",
                   help="+condition Hydra override (NMR input type).")
    p.add_argument("--sources-json", type=Path, default=None,
                   help="Optional JSON file overriding the default 3-source "
                        "cotrain composition (list of {name, hydra_name, "
                        "split_suffix, valid_h_key, valid_c_key, "
                        "dedup_suffix, extra_overrides} dicts -- all but "
                        "name/hydra_name/split_suffix are optional). Use "
                        "this for a different checkpoint/dataset revision "
                        "than the confirmed-working defaults above.")
    p.add_argument("--dedup", action="store_true",
                   help="Use each source's configured `dedup_suffix` "
                        "instead of its training `split_suffix`. Aborts if "
                        "a requested source has no `dedup_suffix` "
                        "configured (see module docstring -- the old "
                        "naive '<suffix>+_dedup' concatenation is no "
                        "longer safe with the new suffix naming).")
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
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                   choices=["train", "val", "test"],
                   help="Which splits to extract.")
    p.add_argument("--ecfp-radius", type=int, default=2)
    p.add_argument("--ecfp-nbits", type=int, default=2048)
    p.add_argument("--save-layer-reps", action="store_true",
                   help="Also save intermediate peak-embedder layer_reps "
                        "(large; off by default).")
    p.add_argument("--save-per-source", action="store_true",
                   help="Additionally save one .pt file per source dataset "
                        "per split (in case you don't want just the merged "
                        "cotrain file).")
    p.add_argument("--n-ecfp-workers", type=int, default=safe_n_workers())
    p.add_argument("--limit", type=int, default=None,
                   help="Debug: cap number of molecules processed per "
                        "(source, split). Do not use for real runs.")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args(argv)

    if args.sources_json is not None:
        sources_raw = json.loads(Path(args.sources_json).read_text())
    else:
        sources_raw = DEFAULT_SOURCES

    sources = [SourceSpec(**s) for s in sources_raw]

    if args.dedup:
        new_sources = []
        for s in sources:
            if s.dedup_suffix is None:
                raise ValueError(
                    f"--dedup was requested but source '{s.name}' has no "
                    f"`dedup_suffix` configured. Naively appending '_dedup' "
                    f"to '{s.split_suffix}' is not safe with the new "
                    f"cotrain-prep suffix naming (see module docstring) -- "
                    f"set the correct zero-shot-eval suffix explicitly via "
                    f"--sources-json."
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

    return RunConfig(
        nmr3d_root=args.nmr3d_root,
        ckpt=args.ckpt,
        ckpt_sigma=resolved_sigma,
        save_dir=args.save_dir,
        prefix=args.prefix,
        condition=args.condition,
        sources=sources,
        dedup=args.dedup,
        splits=args.splits,
        ecfp_radius=args.ecfp_radius,
        ecfp_nbits=args.ecfp_nbits,
        save_layer_reps=args.save_layer_reps,
        save_per_source=args.save_per_source,
        n_ecfp_workers=args.n_ecfp_workers,
        limit=args.limit,
        c_peak_norm_overrides=cpeak_overrides,
        bounds_overrides=bound_overrides,
        device=args.device,
        seed=args.seed,
    )


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------


def load_model(ckpt: Path, device: str, nmr3d_root: Optional[Path] = None):
    """Load the trained NMRTo3DStructureElucidation Lightning module and
    return (model, peak_embedder), following the eval-mode-loading recipe in
    ONBOARDING.md §3.

    The nmr3d import MUST go through `nmr3d_import_scope`: this module's
    top-level `from src.common...` imports have already cached `src` as THIS
    repo's namespace package, so a bare sys.path.insert leaves
    `from src.model.model import ...` raising ModuleNotFoundError."""
    with nmr3d_import_scope(nmr3d_root):
        from src.model.model import NMRTo3DStructureElucidation

        model = NMRTo3DStructureElucidation.load_from_checkpoint(
            str(ckpt), map_location=device
        )
    model.eval()
    model.to(device)

    # The original extraction script indexes through `model.model.score_model`
    # (an extra `.model` level vs. the bare `model.score_model` shown in
    # ONBOARDING.md §3) -- support both so this keeps working if the wrapping
    # changes between checkpoint versions.
    if hasattr(model, "model") and hasattr(model.model, "score_model"):
        score_model = model.model.score_model
    else:
        score_model = model.score_model

    peak_embedder = score_model.y_embedder
    return model, peak_embedder


# -----------------------------------------------------------------------------
# Datamodule construction (one Hydra compose + NMRDataModule per source)
# -----------------------------------------------------------------------------


def build_datamodule(cfg_dir: str, source: "SourceSpec", sigma_data: float, condition: str,
                      nmr3d_root: Optional[Path] = None):
    from hydra import compose, initialize_config_dir

    overrides = [
        f"+data={source.hydra_name}",
        f"+condition={condition}",
        f"dataset_args.split_indices_suffix={source.split_suffix}",
        f"dataset_args.sigma_data={sigma_data}",
        # f"dataset_args.max_n_atoms={200}",
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
            f"The default sources' hydra_name/split_suffix/valid_h_key were "
            f"confirmed working via a --limit 100 test run, so if you're "
            f"using DEFAULT_SOURCES unmodified, the likely cause is the "
            f"(untested) c_peak_norm_args override -- try again without "
            f"--apply-cpeak-norm-overrides to isolate that. If you're using "
            f"a custom --sources-json, double-check that hydra_name exists "
            f"under configs/data/.\n"
            f"Original error: {e}"
        ) from e

    with nmr3d_import_scope(nmr3d_root):
        from src.data.datamodule import NMRDataModule

        dm = NMRDataModule(cfg.dataset_args)
    return dm, cfg


def get_split_dataloader(dm, split: str):
    """NMRDataModule follows the Lightning DataModule fit/test setup
    convention used in enc_extract_embs.py."""
    if split in ("train", "val"):
        dm.prepare_data()
        dm.setup("fit")
        return dm.train_dataloader() if split == "train" else dm.val_dataloader()
    elif split == "test":
        dm.prepare_data()
        dm.setup("test")
        return dm.test_dataloader()
    else:
        raise ValueError(f"Unknown split: {split}")


# -----------------------------------------------------------------------------
# ECFP computation
# -----------------------------------------------------------------------------


def _ecfp_worker(args):
    smiles, radius, n_bits = args

    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    bits = np.zeros(n_bits, dtype=np.uint8)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return bits

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=n_bits,
    )

    fp = generator.GetFingerprint(mol)

    bits[list(fp.GetOnBits())] = 1
    return bits


def compute_ecfps(smiles_list: List[str], radius: int, n_bits: int,
                   n_workers: int) -> torch.Tensor:
    tasks = [(s, radius, n_bits) for s in smiles_list]
    if n_workers <= 1:
        results = [_ecfp_worker(t) for t in tqdm(tasks, desc="ECFP (serial)")]
    else:
        with mp.Pool(n_workers) as pool:
            results = list(
                tqdm(pool.imap(_ecfp_worker, tasks, chunksize=256),
                     total=len(tasks), desc=f"ECFP ({n_workers} workers)")
            )
    return torch.from_numpy(np.stack(results, axis=0))


# -----------------------------------------------------------------------------
# Embedding extraction for one (source, split)
# -----------------------------------------------------------------------------


def extract_source_split(model, peak_embedder, dm, split: str, device: str,
                          save_layer_reps: bool, limit: Optional[int]) -> Dict:
    dataloader = get_split_dataloader(dm, split)

    smiles_out: List[str] = []
    mol_idx_out: List[int] = []
    global_cond_out: List[torch.Tensor] = []
    layer_reps_out: List[torch.Tensor] = []
    have_mol_idx = True

    n_seen = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"  [{split}] forward pass"):
            # Same batch-unpacking convention as enc_extract_embs.py.
            model_inputs, smiles = batch[0]

            condition = {
                k: v.to(device) for k, v in model_inputs["condition"].items()
            }

            global_cond, _, _, layer_reps = peak_embedder(
                condition, extract_all=True,
            )

            global_cond_out.append(global_cond.cpu())
            if save_layer_reps:
                layer_reps_out.append(layer_reps.cpu())
            smiles_out.extend(smiles)

            # mol_idx is global across nmr-to-3d datasets (ONBOARDING.md §1.1)
            # and is very useful for later cross-referencing -- grab it if the
            # datamodule/collate happens to expose it, but don't fail if not.
            if have_mol_idx:
                mol_idx_batch = (
                    model_inputs.get("mol_idx")
                    if isinstance(model_inputs, dict) else None
                )
                if mol_idx_batch is not None:
                    mol_idx_out.extend(
                        mol_idx_batch.detach().cpu().tolist()
                    )
                else:
                    have_mol_idx = False
                    mol_idx_out = []

            n_seen += len(smiles)
            if limit is not None and n_seen >= limit:
                break

    result = {
        "smiles": smiles_out,
        "global_cond": torch.cat(global_cond_out, dim=0),
        "mol_idx": (
            torch.tensor(mol_idx_out, dtype=torch.long)
            if have_mol_idx and len(mol_idx_out) == len(smiles_out)
            else None
        ),
    }
    if save_layer_reps:
        result["layer_reps"] = torch.cat(layer_reps_out, dim=0)
    return result


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> None:
    cfg = parse_args(argv)
    set_seed(cfg.seed)

    sys.path.insert(0, str(cfg.nmr3d_root))
    config_dir = str(cfg.nmr3d_root / "configs")

    cfg.save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("Cotrain embedding extraction (2026-07-20 cotrain-v3 composition)")
    print(f"  ckpt        : {cfg.ckpt}")
    print(f"  ckpt sigma  : {cfg.ckpt_sigma}  (read from the checkpoint)")
    print(f"  condition   : {cfg.condition}")
    print(f"  sources     : {[(s.name, s.hydra_name, s.split_suffix) for s in cfg.sources]}")
    print(f"  splits      : {cfg.splits}")
    print(f"  dedup       : {cfg.dedup}")
    print(f"  save dir    : {cfg.save_dir}")
    print(f"  c_peak_norm : {cfg.c_peak_norm_overrides or '(hydra config defaults)'}")
    print(f"  bounds      : {cfg.bounds_overrides or '(per-source hydra config bounds)'}")
    print("=" * 78)

    model, peak_embedder = load_model(cfg.ckpt, cfg.device, nmr3d_root=cfg.nmr3d_root)

    # split -> accumulated cross-source records
    combined: Dict[str, Dict[str, list]] = {
        split: {"smiles": [], "dataset": [], "mol_idx": [], "global_cond": [], "layer_reps": []}
        for split in cfg.splits
    }
    combined_has_mol_idx = {split: True for split in cfg.splits}

    for source in cfg.sources:
        print(f"\n### Source: {source.name} ({source.hydra_name}, "
              f"suffix={source.split_suffix}, valid_h_key={source.valid_h_key}, "
              f"valid_c_key={source.valid_c_key}) ###")
        dm, _ = build_datamodule(config_dir, source, cfg.ckpt_sigma, cfg.condition,
                                  nmr3d_root=cfg.nmr3d_root)

        for split in cfg.splits:
            out = extract_source_split(
                model, peak_embedder, dm, split, cfg.device,
                cfg.save_layer_reps, cfg.limit,
            )
            n = len(out["smiles"])
            print(f"  [{source.name}/{split}] {n} molecules, "
                  f"global_cond shape={tuple(out['global_cond'].shape)}")

            # Per-source ECFP + save (optional)
            ecfp = compute_ecfps(out["smiles"], cfg.ecfp_radius, cfg.ecfp_nbits,
                                  cfg.n_ecfp_workers)

            if cfg.save_per_source:
                per_source_path = (
                    cfg.save_dir / f"{source.name}_{split}_global_cond.pt"
                )
                torch.save({
                    "split": split,
                    "dataset": source.name,
                    "ckpt": str(cfg.ckpt),
                    "sigma_data": cfg.ckpt_sigma,
                    "condition": cfg.condition,
                    "split_suffix": source.split_suffix,
                    "valid_h_key": source.valid_h_key,
                    "valid_c_key": source.valid_c_key,
                    "smiles": out["smiles"],
                    "mol_idx": out["mol_idx"],
                    "global_cond": out["global_cond"],
                    "ecfp": ecfp,
                    "ecfp_radius": cfg.ecfp_radius,
                    "ecfp_nbits": cfg.ecfp_nbits,
                    **({"layer_reps": out["layer_reps"]} if cfg.save_layer_reps else {}),
                    "extracted_at": datetime.now(timezone.utc).isoformat(),
                    "seed": cfg.seed,
                }, per_source_path)
                print(f"    -> saved {per_source_path}")

            combined[split]["smiles"].extend(out["smiles"])
            combined[split]["dataset"].extend([source.name] * n)
            combined[split]["global_cond"].append(out["global_cond"])
            if cfg.save_layer_reps:
                combined[split]["layer_reps"].append(out["layer_reps"])
            if out["mol_idx"] is not None:
                combined[split]["mol_idx"].append(out["mol_idx"])
            else:
                combined_has_mol_idx[split] = False

    # ---- Merge across sources, compute cotrain-level ECFPs, save ----
    manifest = {
        "ckpt": str(cfg.ckpt),
        "ckpt_sigma": cfg.ckpt_sigma,
        "condition": cfg.condition,
        "sources": [asdict(s) for s in cfg.sources],
        "dedup": cfg.dedup,
        "c_peak_norm_overrides": list(cfg.c_peak_norm_overrides),
        "bounds_overrides": list(cfg.bounds_overrides),
        "ecfp_radius": cfg.ecfp_radius,
        "ecfp_nbits": cfg.ecfp_nbits,
        "seed": cfg.seed,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "outputs": [],
    }

    for split in cfg.splits:
        acc = combined[split]
        n_total = len(acc["smiles"])
        if n_total == 0:
            print(f"[warn] split '{split}' has 0 molecules across all sources, skipping save.")
            continue

        global_cond_all = torch.cat(acc["global_cond"], dim=0)
        mol_idx_all = (
            torch.cat(acc["mol_idx"], dim=0)
            if combined_has_mol_idx[split] and len(acc["mol_idx"]) == len(cfg.sources)
            else None
        )
        # Recompute ECFPs once over the merged SMILES list so the final file
        # is self-contained and doesn't depend on per-source intermediate
        # results (also keeps ordering perfectly aligned with `smiles`/`dataset`).
        ecfp_all = compute_ecfps(acc["smiles"], cfg.ecfp_radius, cfg.ecfp_nbits,
                                  cfg.n_ecfp_workers)

        out_dict = {
            "split": split,
            "ckpt": str(cfg.ckpt),
            "sigma_data": cfg.ckpt_sigma,
            "condition": cfg.condition,
            "sources": [asdict(s) for s in cfg.sources],
            "smiles": acc["smiles"],
            "dataset": acc["dataset"],
            "mol_idx": mol_idx_all,
            "global_cond": global_cond_all,
            "ecfp": ecfp_all,
            "ecfp_radius": cfg.ecfp_radius,
            "ecfp_nbits": cfg.ecfp_nbits,
            "extracted_at": datetime.now(timezone.utc).isoformat(),
            "seed": cfg.seed,
        }
        if cfg.save_layer_reps and acc["layer_reps"]:
            out_dict["layer_reps"] = torch.cat(acc["layer_reps"], dim=0)

        out_path = cfg.save_dir / f"{cfg.prefix}_{split}_global_cond.pt"
        torch.save(out_dict, out_path)
        manifest["outputs"].append({
            "split": split, "path": str(out_path), "n_molecules": n_total,
        })
        print(f"\n✓ Saved {out_path} ({n_total} molecules, "
              f"global_cond shape={tuple(global_cond_all.shape)}, "
              f"ecfp shape={tuple(ecfp_all.shape)})")

    manifest_path = cfg.save_dir / f"{cfg.prefix}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\n✓ Wrote manifest: {manifest_path}")
    print("Done.")


if __name__ == "__main__":
    main()