#!/usr/bin/env python
"""
extract_cotrain_layerwise_embeddings.py
==========================================

Variant of `extract_cotrain_embeddings.py` that additionally extracts
**decoder (trunk) hidden states** at a user-specified set of transformer
layers and diffusion timesteps -- not just the peak-embedder's pooled
`global_cond` -- following the noising/pooling recipe from an earlier
one-off multilayer extraction script.

For each (layer, timestep) combination, this pulls:
    - `x_hidden_mean`: mean-pooled (over valid atoms) trunk hidden state
      for the atom/coordinate stream, at that layer and noise level.
    - `y_hidden_mean`: mean-pooled (over valid NMR tokens) trunk hidden
      state for the NMR-conditioning stream, at that layer and noise level.
`global_cond` (the peak embedder's pooled output) doesn't depend on layer
or timestep, so unlike the reference script it is stored once per molecule
rather than duplicated under every (layer, timestep) key.

Noising matches the diffusion training process: at t<=0.001 the "clean"
coordinates are used directly; at higher t, coordinates are interpolated
toward Gaussian noise via the model's own `interpolate` (flow-matching)
method and re-centered, exactly mirroring what the model sees during
training/sampling at that noise level.

This is *far* more expensive per batch than plain peak-embedder extraction
(one trunk forward pass per (layer, timestep) pair, i.e.
`len(layers) * len(timesteps)` forward passes per batch), so unlike the
plain extraction script this one defaults to a per-source-dataset sample
cap (`--n-samples-per-dataset`, matching the reference script's
`N_SAMPLES=5000`) rather than processing everything.

Outputs (per split)
--------------------
    cotrain_train_layerwise.pt
    cotrain_val_layerwise.pt
    cotrain_test_layerwise.pt

Each is a dict:
    {
        "split", "ckpt", "sigma_data", "condition",
        "layers": [...], "timesteps": [...],
        "smiles": [...], "dataset": [...], "mol_idx": LongTensor or None,
        "global_cond": FloatTensor [N, D],
        "layer_timestep_data": {
            (layer, timestep): {
                "x_hidden_mean": FloatTensor [N, Dx],
                "y_hidden_mean": FloatTensor [N, Dy],
            },
            ...
        },
        "ecfp": ByteTensor [N, n_bits], "ecfp_radius", "ecfp_nbits",
        "extracted_at", "seed",
    }

Usage
-----
python 0a_extr_decoder.py --nmr3d-root /home/jc4587/3_AI4chemistr/nmr-to-3d --ckpt /projects/CRYOEM/zhonglab/data_nmr/2026/ckpts/26-05-01-cotraining-baselines/cotrain-epoch0899-accuracy60_70.ckpt --ckpt-sigma 2.8268 --save-dir /scratch/gpfs/ZHONGE/jc4587/nmr_embs_layerwise --layers -1 0 1 2 3 4 5 6 7 8 9 10 11 --timesteps 0.001 --n-samples-per-dataset 0 --splits train val test
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Default cotrain composition (same as extract_cotrain_embeddings.py)
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
    layers: List[int]
    timesteps: List[float]
    ecfp_radius: int
    ecfp_nbits: int
    save_per_source: bool
    n_ecfp_workers: int
    n_samples_per_dataset: Optional[int]
    device: str
    seed: int


def parse_args(argv: Optional[Sequence[str]] = None) -> RunConfig:
    p = argparse.ArgumentParser(
        description="Extract multi-layer, multi-timestep decoder hidden "
                    "states (+ global_cond, SMILES, ECFP) for the cotrain "
                    "dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--nmr3d-root", type=Path, required=True)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--ckpt-sigma", type=float, required=True)
    p.add_argument("--save-dir", type=Path, required=True)
    p.add_argument("--prefix", type=str, default="cotrain")
    p.add_argument("--condition", type=str, default="hcpeak")
    p.add_argument("--sources-json", type=Path, default=None)
    p.add_argument("--dedup", action="store_true")
    p.add_argument("--splits", nargs="+", default=["train"],
                   choices=["train", "val", "test"])

    p.add_argument("--layers", nargs="+", type=int,
                   default=[-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                   help="Trunk layers to extract hidden states from (-1 == "
                        "final layer, matching the reference script's "
                        "convention).")
    p.add_argument("--timesteps", nargs="+", type=float, default=[0.001, 0.5, 1.0],
                   help="Diffusion timesteps (in [0,1]) to noise coordinates "
                        "to before the trunk forward pass. t<=0.001 uses "
                        "clean coordinates directly.")

    p.add_argument("--ecfp-radius", type=int, default=2)
    p.add_argument("--ecfp-nbits", type=int, default=2048)
    p.add_argument("--save-per-source", action="store_true")
    p.add_argument("--n-ecfp-workers", type=int, default=max(1, mp.cpu_count() - 2))
    p.add_argument("--n-samples-per-dataset", type=int, default=5000,
                   help="Cap on molecules processed per source dataset per "
                        "split -- this extraction runs len(layers)*"
                        "len(timesteps) trunk forward passes per batch, so "
                        "full-scale runs are typically infeasible. Pass a "
                        "large number (or 0) to disable the cap.")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args(argv)

    sources_raw = json.loads(Path(args.sources_json).read_text()) if args.sources_json else DEFAULT_SOURCES
    sources = [SourceSpec(**s) for s in sources_raw]
    if args.dedup:
        sources = [SourceSpec(s.name, s.hydra_name, s.split_suffix + "_dedup") for s in sources]

    n_cap = None if (args.n_samples_per_dataset is None or args.n_samples_per_dataset <= 0) else args.n_samples_per_dataset

    return RunConfig(
        nmr3d_root=args.nmr3d_root, ckpt=args.ckpt, ckpt_sigma=args.ckpt_sigma,
        save_dir=args.save_dir, prefix=args.prefix, condition=args.condition,
        sources=sources, dedup=args.dedup, splits=args.splits,
        layers=args.layers, timesteps=args.timesteps,
        ecfp_radius=args.ecfp_radius, ecfp_nbits=args.ecfp_nbits,
        save_per_source=args.save_per_source, n_ecfp_workers=args.n_ecfp_workers,
        n_samples_per_dataset=n_cap, device=args.device, seed=args.seed,
    )


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------


def load_model(ckpt: Path, device: str):
    """Returns (model, trunk, peak_embedder, diffusion_module).

    `trunk` is the score-model transformer (callable with
    `extract_features=True, extract_layer=...`); `diffusion_module` is
    whatever object exposes `.interpolate` / `._apply_coordinate_transform`
    (the flow-matching noising utilities) -- both are resolved with a
    fallback in case the Lightning module wrapping differs across
    checkpoint versions (mirrors extract_cotrain_embeddings.py's approach).
    """
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
# Datamodule construction (identical to extract_cotrain_embeddings.py)
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
    return NMRDataModule(cfg.dataset_args)


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


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask[..., None].float()
    return (x * mask).sum(1) / mask.sum(1).clamp(min=1)


# -----------------------------------------------------------------------------
# ECFP computation (same as extract_cotrain_embeddings.py)
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
                                 device: str, n_samples_cap: Optional[int]) -> Dict:
    dataloader = get_split_dataloader(dm, split)

    smiles_out: List[str] = []
    mol_idx_out: List[int] = []
    # global_cond_out: List[torch.Tensor] = []
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

            # global_cond, _, _ = peak_embedder(condition)
            # global_cond_out.append(global_cond.cpu())

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
            if have_mol_idx and len(mol_idx_out) == n_final
            else None
        ),
        "layer_timestep_data": {
            key: {
                name: torch.cat(vals, dim=0)[:n_final]
                for name, vals in val.items()
            }
            for key, val in layer_timestep_out.items()
        },
    }
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
    print("Cotrain layerwise embedding extraction")
    print(f"  ckpt        : {cfg.ckpt}")
    print(f"  ckpt sigma  : {cfg.ckpt_sigma}")
    print(f"  condition   : {cfg.condition}")
    print(f"  sources     : {[s.name for s in cfg.sources]}")
    print(f"  splits      : {cfg.splits}")
    print(f"  layers      : {cfg.layers}")
    print(f"  timesteps   : {cfg.timesteps}")
    print(f"  n_samples/ds: {cfg.n_samples_per_dataset or 'ALL (expensive!)'}")
    print("=" * 78)

    model, trunk, peak_embedder, diffusion_module = load_model(cfg.ckpt, cfg.device)

    combined: Dict[str, Dict] = {
        split: {
            "smiles": [], "dataset": [], "mol_idx": [], 
            "layer_timestep_data": {(l, t): {"x_hidden_mean": [], "y_hidden_mean": []}
                                     for l in cfg.layers for t in cfg.timesteps},
        }
        for split in cfg.splits
    }
    combined_has_mol_idx = {split: True for split in cfg.splits}

    for source in cfg.sources:
        print(f"\n### Source: {source.name} ({source.hydra_name}, suffix={source.split_suffix}) ###")
        dm = build_datamodule(config_dir, source.hydra_name, source.split_suffix, cfg.ckpt_sigma, cfg.condition)

        for split in cfg.splits:
            out = extract_layerwise_for_split(
                model, trunk, peak_embedder, diffusion_module, dm, split,
                cfg.layers, cfg.timesteps, cfg.device, cfg.n_samples_per_dataset,
            )
            n = len(out["smiles"])
            print(f"  [{source.name}/{split}] {n} molecules, ")

            # ecfp = compute_ecfps(out["smiles"], cfg.ecfp_radius, cfg.ecfp_nbits, cfg.n_ecfp_workers)

            if cfg.save_per_source:
                per_source_path = cfg.save_dir / f"{source.name}_{split}_layerwise.pt"
                torch.save({
                    "split": split, "dataset": source.name, "ckpt": str(cfg.ckpt),
                    "sigma_data": cfg.ckpt_sigma, "condition": cfg.condition,
                    "layers": cfg.layers, "timesteps": cfg.timesteps,
                    "split_suffix": source.split_suffix,
                    "smiles": out["smiles"], "mol_idx": out["mol_idx"],
                    # "global_cond": out["global_cond"],
                    "layer_timestep_data": out["layer_timestep_data"],
                    # "ecfp": ecfp, "ecfp_radius": cfg.ecfp_radius, "ecfp_nbits": cfg.ecfp_nbits,
                    "extracted_at": datetime.now(timezone.utc).isoformat(), "seed": cfg.seed,
                }, per_source_path)
                print(f"    -> saved {per_source_path}")

            combined[split]["smiles"].extend(out["smiles"])
            combined[split]["dataset"].extend([source.name] * n)
            # combined[split]["global_cond"].append(out["global_cond"])
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
        "layers": cfg.layers, "timesteps": cfg.timesteps,
        # "ecfp_radius": cfg.ecfp_radius, "ecfp_nbits": cfg.ecfp_nbits,
        "n_samples_per_dataset": cfg.n_samples_per_dataset, "seed": cfg.seed,
        "extracted_at": datetime.now(timezone.utc).isoformat(), "outputs": [],
    }

    for split in cfg.splits:
        acc = combined[split]
        n_total = len(acc["smiles"])
        if n_total == 0:
            print(f"[warn] split '{split}' has 0 molecules across all sources, skipping save.")
            continue

        # global_cond_all = torch.cat(acc["global_cond"], dim=0)
        mol_idx_all = (
            torch.cat(acc["mol_idx"], dim=0)
            if combined_has_mol_idx[split] and len(acc["mol_idx"]) == len(cfg.sources) else None
        )
        # ecfp_all = compute_ecfps(acc["smiles"], cfg.ecfp_radius, cfg.ecfp_nbits, cfg.n_ecfp_workers)

        layer_timestep_all = {
            key: {name: torch.cat(vals, dim=0) for name, vals in val.items()}
            for key, val in acc["layer_timestep_data"].items()
        }

        out_dict = {
            "split": split, "ckpt": str(cfg.ckpt), "sigma_data": cfg.ckpt_sigma,
            "condition": cfg.condition, "layers": cfg.layers, "timesteps": cfg.timesteps,
            "sources": [asdict(s) for s in cfg.sources],
            "smiles": acc["smiles"], "dataset": acc["dataset"], "mol_idx": mol_idx_all,
            # "global_cond": global_cond_all, "layer_timestep_data": layer_timestep_all,
            # "ecfp": ecfp_all, "ecfp_radius": cfg.ecfp_radius, "ecfp_nbits": cfg.ecfp_nbits,
            "extracted_at": datetime.now(timezone.utc).isoformat(), "seed": cfg.seed,
        }

        out_path = cfg.save_dir / f"{cfg.prefix}_{split}_layerwise.pt"
        torch.save(out_dict, out_path)
        manifest["outputs"].append({"split": split, "path": str(out_path), "n_molecules": n_total})
        print(f"\n✓ Saved {out_path} ({n_total} molecules, "
              f"{len(layer_timestep_all)} (layer,timestep) combos, ")

    manifest_path = cfg.save_dir / f"{cfg.prefix}_layerwise_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\n✓ Wrote manifest: {manifest_path}")
    print("Done.")


if __name__ == "__main__":
    main()
