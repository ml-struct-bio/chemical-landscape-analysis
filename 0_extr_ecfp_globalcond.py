#!/usr/bin/env python
"""
0_extr_ecfp_globalcond.py
==============================

End-to-end, checkpoint-agnostic extraction of:
    - SMILES
    - NMR peak-embedder ("global_cond") embeddings
    - ECFP (Morgan) fingerprints
    - source-dataset labels

for every molecule in the *cotrain* dataset
(`cotrain-nmrexp-nmrpeak-spectranp-uspto` = nmrexp-nmrpeak + spectranp-760k +
uspto), for the train / val / test splits.

Why per-source dataloaders instead of the joint `cotrain-*` Hydra config
--------------------------------------------------------------------------
The joint `+data=cotrain-nmrexp-nmrpeak-spectranp-uspto` config drives
`SourceWeightedDistributedSampler`, which is a *weighted, categorical*
sampler meant for training (it samples with replacement / reweighting, not
"iterate every molecule exactly once"). For embedding extraction we want a
full, deterministic pass over every molecule in each member dataset, so this
script instantiates one `NMRDataModule` per source dataset (same pattern as
the original `enc_extract_embs.py`) and simply tags each molecule with its
source name. The three sources are then concatenated back together into a
single "cotrain" train/val/test dict, per the request.

Reproducibility / generalization notes (see CLAUDE.md + ONBOARDING.md)
--------------------------------------------------------------------------
- `sigma_data` for any cotrain checkpoint is a *computed* weighted-RMS
  mixture over the three sources (baked into the checkpoint's training run,
  e.g. 2.8268 for the equal-weighted 2026-05-01 milestone), NOT any single
  dataset's own `sigma_data` default. This must be supplied per-checkpoint
  via `--ckpt-sigma`; the script does not guess it.
- Val/test splits are restricted to molecules with BOTH H1 and C13 spectra
  regardless of the `split_indices_suffix` used for training (see
  ONBOARDING.md §1.4), so the same suffix is reused across train/val/test.
- Use `--dedup` to switch to the zero-shot dedup splits (`_or_dedup` /
  `_both_orig_dedup`) instead of the raw training splits.
- `mol_idx` is global across all nmr-to-3d datasets (ONBOARDING.md §1.1), so
  it is saved whenever the datamodule/dataset exposes it, to make later
  cross-dataset / cross-checkpoint joins trivial.

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
        "sources": [{"name", "hydra_name", "split_suffix"}, ...],
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
python 0_extr_ecfp_globalcond.py --nmr3d-root /home/jc4587/3_AI4chemistr/nmr-to-3d --ckpt /projects/CRYOEM/zhonglab/data_nmr/2026/ckpts/26-05-01-cotraining-baselines/cotrain-epoch0899-accuracy60_70.ckpt --ckpt-sigma 2.8268 --save-dir /scratch/gpfs/ZHONGE/jc4587/nmr_embs_FINAL --prefix cotrain
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from rdkit.Chem import rdFingerprintGenerator
import numpy as np
import torch
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Default cotrain composition (ONBOARDING.md §1.3 / §2.1)
# -----------------------------------------------------------------------------

DEFAULT_SOURCES = [
    {"name": "nmrexp", "hydra_name": "nmrexp-nmrpeak", "split_suffix": "_or"},
    {"name": "spectranp", "hydra_name": "spectranp-760k", "split_suffix": "_or"},
    {"name": "uspto", "hydra_name": "uspto", "split_suffix": "_both_orig"},
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
    p.add_argument("--ckpt-sigma", type=float, required=True,
                   help="sigma_data used when this checkpoint was trained "
                        "(the cotrain-mixture value, e.g. 2.8268 for the "
                        "2026-05-01 equal-weighted milestone -- NOT a "
                        "per-dataset default; see ONBOARDING.md §2.1).")
    p.add_argument("--save-dir", type=Path, required=True,
                   help="Directory to write output .pt files + manifest.")
    p.add_argument("--prefix", type=str, default="cotrain",
                   help="Filename prefix, e.g. '<prefix>_train_global_cond.pt'.")
    p.add_argument("--condition", type=str, default="hcpeak",
                   help="+condition Hydra override (NMR input type).")
    p.add_argument("--sources-json", type=Path, default=None,
                   help="Optional JSON file overriding the default 3-source "
                        "cotrain composition (list of "
                        "{name, hydra_name, split_suffix} dicts).")
    p.add_argument("--dedup", action="store_true",
                   help="Use zero-shot dedup splits (append '_dedup' to each "
                        "source's split_suffix) instead of the raw splits.")
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
    p.add_argument("--n-ecfp-workers", type=int, default=max(1, mp.cpu_count() - 2))
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
        sources = [
            SourceSpec(s.name, s.hydra_name, s.split_suffix + "_dedup")
            for s in sources
        ]

    return RunConfig(
        nmr3d_root=args.nmr3d_root,
        ckpt=args.ckpt,
        ckpt_sigma=args.ckpt_sigma,
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
        device=args.device,
        seed=args.seed,
    )


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------


def load_model(ckpt: Path, device: str):
    """Load the trained NMRTo3DStructureElucidation Lightning module and
    return (model, peak_embedder), following the eval-mode-loading recipe in
    ONBOARDING.md §3."""
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


def build_datamodule(cfg_dir: str, hydra_name: str, split_suffix: str,
                      sigma_data: float, condition: str):
    from hydra import compose, initialize_config_dir
    from src.data.datamodule import NMRDataModule

    with initialize_config_dir(cfg_dir, version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                f"+data={hydra_name}",
                f"+condition={condition}",
                f"dataset_args.split_indices_suffix={split_suffix}",
                f"dataset_args.sigma_data={sigma_data}",
            ],
        )

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
    print("Cotrain embedding extraction")
    print(f"  ckpt        : {cfg.ckpt}")
    print(f"  ckpt sigma  : {cfg.ckpt_sigma}")
    print(f"  condition   : {cfg.condition}")
    print(f"  sources     : {[s.name for s in cfg.sources]}")
    print(f"  splits      : {cfg.splits}")
    print(f"  dedup       : {cfg.dedup}")
    print(f"  save dir    : {cfg.save_dir}")
    print("=" * 78)

    model, peak_embedder = load_model(cfg.ckpt, cfg.device)

    # split -> accumulated cross-source records
    combined: Dict[str, Dict[str, list]] = {
        split: {"smiles": [], "dataset": [], "mol_idx": [], "global_cond": [], "layer_reps": []}
        for split in cfg.splits
    }
    combined_has_mol_idx = {split: True for split in cfg.splits}

    for source in cfg.sources:
        print(f"\n### Source: {source.name} ({source.hydra_name}, "
              f"suffix={source.split_suffix}) ###")
        dm, _ = build_datamodule(
            config_dir, source.hydra_name, source.split_suffix,
            cfg.ckpt_sigma, cfg.condition,
        )

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