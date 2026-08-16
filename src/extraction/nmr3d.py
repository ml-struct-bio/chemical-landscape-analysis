"""Talking to the nmr-to-3d model repo: import scoping, checkpoint loading, and
hydra datamodule construction.

Lifted verbatim from the previous pipeline's
`src/analysis/real_vs_synthetic_analysis.py` (lines 62-167). It lives under
`src/extraction/` now because that is what it actually is -- the machinery for
getting data and models out of the training repo -- and because importing it
from the old location dragged in matplotlib, umap and the palette module as a
side effect, none of which extraction needs.
"""
from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from typing import Optional, Sequence


@contextlib.contextmanager
def nmr3d_import_scope(nmr3d_root: Optional[Path]):
    """Makes `from src.xxx import yyy` resolve against nmr3d_root's OWN
    `src` package for the duration of the `with` block, then restores this
    repo's `src.*` modules afterward.

    Necessary because this repo's `src/` has no `__init__.py` (an implicit
    PEP 420 namespace package), while nmr-to-3d's `src/` is a regular
    package (has `__init__.py`). The moment anything in this module does
    `from src.analysis...`/`from src.common...` (which always happens at
    import time, before nmr3d_root is ever known), Python permanently caches
    `src` in sys.modules as OUR namespace package. Simply prepending
    nmr3d_root to sys.path afterward does NOT make that cached object
    resolve submodules against nmr3d_root -- `from src.model.model import
    ...` then fails with `ModuleNotFoundError: No module named 'src.model'`
    even though nmr3d_root/src/model/model.py exists. Evicting the cached
    `src`/`src.*` entries forces Python to re-resolve `src` from scratch
    against the current sys.path (now with nmr3d_root first), which finds
    nmr3d_root's REGULAR package instead."""
    if nmr3d_root is None:
        yield
        return

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


def load_model(ckpt: Path, device: str, nmr3d_root: Optional[Path] = None):
    with nmr3d_import_scope(nmr3d_root):
        from src.model.model import NMRTo3DStructureElucidation

        model = NMRTo3DStructureElucidation.load_from_checkpoint(str(ckpt), map_location=device)
        model.eval()
        model.to(device)
        if hasattr(model, "model") and hasattr(model.model, "score_model"):
            score_model = model.model.score_model
        else:
            score_model = model.score_model
        peak_embedder = score_model.y_embedder
    return model, peak_embedder


def build_datamodule(cfg_dir: str, hydra_name: str, split_suffix: Optional[str], sigma_data: float, condition: str,
                      nmr3d_root: Optional[Path] = None,
                      c_peak_norm_overrides: Optional[Sequence[str]] = None,
                      extra_overrides: Optional[Sequence[str]] = None):
    """`split_suffix=None` skips the `dataset_args.split_indices_suffix` override
    entirely, so the hydra data config's own baked-in default (e.g.
    `real-famous-np.yaml`'s `split_indices_suffix: "_or"`) is used as-is --
    useful for one-off `real-*` datasets that already specify a sensible
    default rather than being part of a larger split-suffix-per-source
    cotrain mixture.

    `c_peak_norm_overrides` pins `dataset_args.c_peak_norm_args.*` to whatever
    the checkpoint was trained with (see `src/common/ckpt_meta.py`). This one
    is NOT optional in the way `sigma_data` is: the 13C symmetry-merge is a
    normalization that `augment_c_peaks` applies at val/test too, so leaving
    it at the hydra config's default silently feeds the model spectra it never
    saw whenever the ckpt trained with a different `dedup_p`. Passing None
    keeps the config default, which is only right for a `dedup_p: 0.0`
    checkpoint.

    `extra_overrides` is a free-form escape hatch for any further
    `dataset_args.*` hydra override a caller needs -- e.g. the mixture-wide
    padding bounds from `ckpt_meta.bounds_overrides()` when re-extracting the
    cotrain training sources themselves. Analyses that embed HELD-OUT datasets
    deliberately do NOT
    apply those, since truncating held-out datasets to the training mixture's
    peak counts would drop real peaks)."""
    with nmr3d_import_scope(nmr3d_root):
        from hydra import compose, initialize_config_dir
        from src.data.datamodule import NMRDataModule

        overrides = [f"+data={hydra_name}", f"+condition={condition}", f"dataset_args.sigma_data={sigma_data}"]
        if split_suffix is not None:
            overrides.append(f"dataset_args.split_indices_suffix={split_suffix}")
        overrides.extend(c_peak_norm_overrides or [])
        overrides.extend(extra_overrides or [])

        with initialize_config_dir(cfg_dir, version_base=None):
            cfg = compose(config_name="config", overrides=overrides)
        dm = NMRDataModule(cfg.dataset_args)
    return dm


def get_split_dataloader(dm, split: str):
    if split in ("train", "val"):
        dm.prepare_data()
        dm.setup("fit")
        return dm.train_dataloader() if split == "train" else dm.val_dataloader()
    if split == "test":
        dm.prepare_data()
        dm.setup("test")
        return dm.test_dataloader()
    raise ValueError(f"Unknown split: {split}")
