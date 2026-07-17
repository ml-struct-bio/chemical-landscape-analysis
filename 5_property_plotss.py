#!/usr/bin/env python
"""
umap_splits_and_properties.py
================================

Fits UMAP on the TRAIN split only, projects val/test into that same fitted
manifold (`reducer.transform`, not a fresh fit -- so all three splits live
in one consistent 2D space), then:

    1. Visualizes the splits together (one overlay plot) and separately
       (one panel per split), colored by source dataset.
    2. Quantifies train/val/test distribution shift in a text report:
       high-dim embedding centroid shift, 2D UMAP centroid shift,
       KS-tests per molecular property, and a nearest-neighbor "domain
       gap" ratio (mean val/test-to-train NN distance vs. train's own
       internal NN distance).
    3. Renders a series of UMAP scatter plots continuously colored by a
       fixed panel of molecular properties (defined in-code, as requested),
       for whichever dataset/split subset you choose, in the exact visual
       style of `umap_main_figure.py` (figsize, marker size/alpha, dpi,
       axis labels) but with a colorbar instead of a discrete legend.

Usage
-----
    python umap_splits_and_properties.py \\
        --data-dir /scratch/gpfs/ZHONGE/jc4587/nmr_embs_cotrain \\
        --prefix cotrain \\
        --out-dir umap_splits_summary \\
        --property-datasets spectranp nmrexp uspto \\
        --property-splits train \\
        --cmap viridis
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import Descriptors

# -----------------------------------------------------------------------------
# Fixed property panel (defined in-code, as requested)
# -----------------------------------------------------------------------------


def _aromatic_fraction(mol) -> float:
    if mol.GetNumAtoms() == 0:
        return 0.0
    return sum(1 for a in mol.GetAtoms() if a.GetIsAromatic()) / mol.GetNumAtoms()


PROPERTIES = {
    "MolWt": Descriptors.MolWt,
    "LogP": Descriptors.MolLogP,
    "TPSA": Descriptors.TPSA,
    "RingCount": Descriptors.RingCount,
    "Aromaticity": _aromatic_fraction,
}

QUALITATIVE_PALETTE = plt.cm.tab10(np.linspace(0, 1, 10))

# Same plot-style constants as umap_main_figure.py, kept identical on purpose.
FIGSIZE = (10, 10)
POINT_SIZE = 0.25
POINT_ALPHA = 0.25
DPI = 600


# -----------------------------------------------------------------------------
# Data loading (per-split, so train/val/test stay distinguishable)
# -----------------------------------------------------------------------------


def load_split(data_dir: Path, prefix: str, split: str, embedding_key: str
                ) -> Optional[Dict]:
    path = data_dir / f"{prefix}_{split}_global_cond.pt"
    if not path.exists():
        print(f"[warn] {path} not found -- skipping split '{split}'.")
        return None
    print(f"Loading {path} ...")
    d = torch.load(path, map_location="cpu")
    if embedding_key not in d:
        raise KeyError(f"'{embedding_key}' not found in {path}. Keys: {list(d.keys())}")
    return {
        "smiles": d["smiles"],
        "dataset": np.asarray(d["dataset"]),
        "embedding": d[embedding_key].numpy().astype(np.float32),
    }


def filter_by_datasets(split_data: Dict, datasets: Optional[Sequence[str]]) -> Dict:
    if datasets is None or split_data is None:
        return split_data
    mask = np.isin(split_data["dataset"], list(datasets))
    return {
        "smiles": [s for s, m in zip(split_data["smiles"], mask) if m],
        "dataset": split_data["dataset"][mask],
        "embedding": split_data["embedding"][mask],
    }


# -----------------------------------------------------------------------------
# UMAP: fit on train, transform val/test
# -----------------------------------------------------------------------------


def fit_umap_on_train(embedding_train: np.ndarray, n_neighbors: int, min_dist: float,
                       metric: str, seed: int, save_model_path: Optional[Path],
                       load_model_path: Optional[Path]):
    if load_model_path is not None:
        print(f"Loading fitted UMAP reducer from {load_model_path}")
        with open(load_model_path, "rb") as f:
            reducer = pickle.load(f)
        return reducer, reducer.embedding_

    import umap.umap_ as umap_lib

    print(f"Fitting UMAP on TRAIN ({embedding_train.shape[0]} points, "
          f"n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}) ...")
    reducer = umap_lib.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                             metric=metric, random_state=seed)
    emb2d_train = reducer.fit_transform(embedding_train)

    if save_model_path is not None:
        save_model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_model_path, "wb") as f:
            pickle.dump(reducer, f)
        print(f"✓ Cached fitted UMAP reducer to {save_model_path}")

    return reducer, emb2d_train


def transform_split(reducer, embedding: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if embedding is None or embedding.shape[0] == 0:
        return None
    return reducer.transform(embedding)


# -----------------------------------------------------------------------------
# Plot helpers (shared style with umap_main_figure.py)
# -----------------------------------------------------------------------------


def color_by_dataset(dataset_labels: np.ndarray) -> Tuple[np.ndarray, Dict[str, tuple]]:
    sources = sorted(set(dataset_labels.tolist()))
    palette = {src: QUALITATIVE_PALETTE[i % len(QUALITATIVE_PALETTE)] for i, src in enumerate(sources)}
    colors = np.array([palette[s] for s in dataset_labels])
    return colors, palette


def plot_splits_combined(emb2d_by_split: Dict[str, np.ndarray], dataset_by_split: Dict[str, np.ndarray],
                          out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Draw train dimmer (usually far larger N) so val/test aren't swamped.
    split_alpha = {"train": POINT_ALPHA * 0.6, "val": POINT_ALPHA * 2, "test": POINT_ALPHA * 2}
    split_size = {"train": POINT_SIZE, "val": POINT_SIZE * 4, "test": POINT_SIZE * 4}

    palette = None
    for split in ["train", "val", "test"]:
        if split not in emb2d_by_split:
            continue
        colors, palette = color_by_dataset(dataset_by_split[split])
        ax.scatter(emb2d_by_split[split][:, 0], emb2d_by_split[split][:, 1],
                   c=colors, s=split_size[split], alpha=split_alpha[split], linewidths=0)

    if palette is not None:
        handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                               markersize=8, label=src) for src, c in palette.items()]
        ax.legend(handles=handles, loc="best", frameon=True, fontsize=9)

    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title("All splits (train dim, val/test emphasized), colored by dataset")
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved {out_path}")


def plot_split_separately(split: str, emb2d: np.ndarray, dataset_labels: np.ndarray,
                           xlim: Tuple[float, float], ylim: Tuple[float, float],
                           out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    colors, palette = color_by_dataset(dataset_labels)
    ax.scatter(emb2d[:, 0], emb2d[:, 1], c=colors, s=POINT_SIZE, alpha=POINT_ALPHA, linewidths=0)

    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                           markersize=8, label=src) for src, c in palette.items()]
    ax.legend(handles=handles, loc="best", frameon=True, fontsize=9)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title(f"{split} split, colored by dataset")
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved {out_path}")


def plot_property_colored(emb2d: np.ndarray, values: np.ndarray, prop_name: str,
                           cmap: str, out_path: Path) -> None:
    mask = np.isfinite(values)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sc = ax.scatter(emb2d[mask, 0], emb2d[mask, 1], c=values[mask], cmap=cmap,
                     s=POINT_SIZE, alpha=POINT_ALPHA, linewidths=0)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label(prop_name)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title(f"Colored by {prop_name}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved {out_path}")


# -----------------------------------------------------------------------------
# Properties (parallel computation)
# -----------------------------------------------------------------------------


def _property_worker(smi: str) -> List[float]:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return [np.nan] * len(PROPERTIES)
    vals = []
    for fn in PROPERTIES.values():
        try:
            vals.append(float(fn(mol)))
        except Exception:
            vals.append(np.nan)
    return vals


def compute_property_matrix(smiles_list: List[str], n_workers: int) -> np.ndarray:
    if n_workers <= 1:
        rows = [_property_worker(s) for s in tqdm(smiles_list, desc="Properties")]
    else:
        with mp.Pool(n_workers) as pool:
            rows = list(tqdm(pool.imap(_property_worker, smiles_list, chunksize=256),
                              total=len(smiles_list), desc=f"Properties ({n_workers} workers)"))
    return np.array(rows, dtype=np.float64)


# -----------------------------------------------------------------------------
# Distribution-shift quantification
# -----------------------------------------------------------------------------


def chunked_nn_dist_cross(query_x: np.ndarray, ref_x: np.ndarray, device: str,
                           chunk_size: int = 2000) -> np.ndarray:
    """For each row in query_x, nearest-neighbor Euclidean distance to ref_x."""
    ref = torch.from_numpy(ref_x).to(device)
    out = np.empty(query_x.shape[0], dtype=np.float32)
    for start in range(0, query_x.shape[0], chunk_size):
        end = min(start + chunk_size, query_x.shape[0])
        q = torch.from_numpy(query_x[start:end]).to(device)
        d = torch.cdist(q, ref)
        out[start:end] = d.min(dim=1).values.cpu().numpy()
    return out


def chunked_nn_dist_within(x: np.ndarray, device: str, chunk_size: int = 2000) -> np.ndarray:
    """For each row in x, nearest-neighbor distance to the rest of x (self excluded)."""
    ref = torch.from_numpy(x).to(device)
    out = np.empty(x.shape[0], dtype=np.float32)
    n = x.shape[0]
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        q = torch.from_numpy(x[start:end]).to(device)
        d = torch.cdist(q, ref)
        for i, global_i in enumerate(range(start, end)):
            d[i, global_i] = float("inf")
        out[start:end] = d.min(dim=1).values.cpu().numpy()
    return out


def quantify_shift_for_group(label: str, train_emb: np.ndarray, other_embs: Dict[str, np.ndarray],
                              train_2d: np.ndarray, other_2ds: Dict[str, np.ndarray],
                              train_props: np.ndarray, other_props: Dict[str, np.ndarray],
                              property_names: List[str], nn_train_sample: int, nn_eval_sample: int,
                              device: str, rng: np.random.Generator) -> List[str]:
    lines = [f"--- {label} ---"]

    train_mean = train_emb.mean(axis=0)
    train_std_norm = float(np.linalg.norm(train_emb.std(axis=0)))

    train_2d_mean = train_2d.mean(axis=0)

    # NN reference sample (train)
    n_ref = min(nn_train_sample, train_emb.shape[0])
    ref_idx = rng.choice(train_emb.shape[0], size=n_ref, replace=False)
    train_ref = train_emb[ref_idx]
    train_self_nn = chunked_nn_dist_within(train_ref, device)
    train_self_nn_mean = float(train_self_nn.mean())

    for split_name, emb in other_embs.items():
        if emb is None or emb.shape[0] == 0:
            continue

        centroid_shift = float(np.linalg.norm(emb.mean(axis=0) - train_mean))
        centroid_shift_norm = centroid_shift / (train_std_norm + 1e-12)

        emb2d = other_2ds[split_name]
        centroid_shift_2d = float(np.linalg.norm(emb2d.mean(axis=0) - train_2d_mean))

        n_eval = min(nn_eval_sample, emb.shape[0])
        eval_idx = rng.choice(emb.shape[0], size=n_eval, replace=False)
        cross_nn = chunked_nn_dist_cross(emb[eval_idx], train_ref, device)
        cross_nn_mean = float(cross_nn.mean())
        domain_gap_ratio = cross_nn_mean / (train_self_nn_mean + 1e-12)

        lines.append(f"  [train vs. {split_name}]")
        lines.append(f"    high-dim embedding centroid shift        : {centroid_shift:.4f} "
                     f"(normalized by train std-norm: {centroid_shift_norm:.4f})")
        lines.append(f"    2D UMAP centroid shift                   : {centroid_shift_2d:.4f}")
        lines.append(f"    mean NN dist, {split_name}->train (n={n_eval:,}) : {cross_nn_mean:.4f}")
        lines.append(f"    mean NN dist, train->train (n={n_ref:,})       : {train_self_nn_mean:.4f}")
        lines.append(f"    domain-gap ratio ({split_name}->train / train->train): {domain_gap_ratio:.3f} "
                     f"(>1 means {split_name} sits further from the train manifold than train "
                     f"points sit from each other)")

        props_other = other_props[split_name]
        for p_idx, p_name in enumerate(property_names):
            a = train_props[:, p_idx]
            b = props_other[:, p_idx]
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            if len(a) < 10 or len(b) < 10:
                continue
            stat, pval = ks_2samp(a, b)
            lines.append(f"    KS test [{p_name}] train vs {split_name}: "
                         f"D={stat:.4f}, p={pval:.3g}")
        lines.append("")

    return lines


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fit UMAP on train, project val/test, visualize + "
                     "quantify split shift, and render property-colored "
                     "UMAP figures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--prefix", type=str, default="cotrain")
    p.add_argument("--embedding-key", type=str, default="global_cond")
    p.add_argument("--out-dir", type=Path, default=Path("umap_splits_summary"))

    p.add_argument("--datasets", nargs="+", default=None,
                   help="Restrict everything in this script (fit, split "
                        "plots, shift stats, property plots) to these "
                        "source datasets. Default: use everything present.")

    # UMAP
    p.add_argument("--umap-n-neighbors", type=int, default=30)
    p.add_argument("--umap-min-dist", type=float, default=0.1)
    p.add_argument("--umap-metric", type=str, default="cosine")
    p.add_argument("--save-umap-model", type=Path, default=None,
                   help="Pickle the fitted UMAP reducer here for reuse.")
    p.add_argument("--load-umap-model", type=Path, default=None,
                   help="Load a previously pickled fitted-on-train reducer "
                        "instead of fitting fresh.")

    # Shift quantification
    p.add_argument("--nn-train-sample", type=int, default=20_000,
                   help="Train reference sample size for NN-based domain-gap stats.")
    p.add_argument("--nn-eval-sample", type=int, default=2_000,
                   help="Val/test sample size for NN-based domain-gap stats.")
    p.add_argument("--n-prop-workers", type=int, default=max(1, mp.cpu_count() - 2))

    # Property-colored plots
    p.add_argument("--property-splits", nargs="+", default=["train"],
                   choices=["train", "val", "test"],
                   help="Which split(s) to combine for the property-colored figures.")
    p.add_argument("--property-datasets", nargs="+", default=None,
                   help="Restrict the property-colored figures to these "
                        "source datasets (spectranp/nmrexp/uspto/any "
                        "subset). Default: same as --datasets.")
    p.add_argument("--cmap", type=str, default="viridis",
                   help="Matplotlib colormap for the property-colored figures.")
    p.add_argument("--property-sample-size", type=int, default=None,
                   help="Optional random cap on molecules used for the "
                        "property figures (descriptor computation is the "
                        "bottleneck at full scale). Default: use all.")

    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("UMAP split visualization + shift quantification + property maps")
    print(f"  data dir : {args.data_dir}")
    print(f"  datasets : {args.datasets or 'ALL'}")
    print("=" * 78)

    # ---- Load + filter splits ----
    raw = {s: load_split(args.data_dir, args.prefix, s, args.embedding_key)
           for s in ["train", "val", "test"]}
    if raw["train"] is None:
        raise FileNotFoundError("A 'train' split is required to fit UMAP.")

    splits = {s: filter_by_datasets(d, args.datasets) for s, d in raw.items() if d is not None}

    for s, d in splits.items():
        print(f"  {s}: {len(d['smiles'])} molecules")

    # ---- Fit UMAP on train, transform val/test ----
    reducer, emb2d_train = fit_umap_on_train(
        splits["train"]["embedding"], args.umap_n_neighbors, args.umap_min_dist,
        args.umap_metric, args.seed, args.save_umap_model, args.load_umap_model,
    )
    emb2d_by_split = {"train": emb2d_train}
    for s in ["val", "test"]:
        if s in splits:
            emb2d_by_split[s] = transform_split(reducer, splits[s]["embedding"])

    dataset_by_split = {s: splits[s]["dataset"] for s in emb2d_by_split}

    # ---- Part 1/2: split visualization (combined + separate) ----
    plot_splits_combined(emb2d_by_split, dataset_by_split,
                          args.out_dir / f"{args.prefix}_splits_combined.png")

    all_x = np.concatenate([e[:, 0] for e in emb2d_by_split.values()])
    all_y = np.concatenate([e[:, 1] for e in emb2d_by_split.values()])
    xlim = (all_x.min(), all_x.max())
    ylim = (all_y.min(), all_y.max())

    for s, emb2d in emb2d_by_split.items():
        plot_split_separately(s, emb2d, dataset_by_split[s], xlim, ylim,
                               args.out_dir / f"{args.prefix}_split_{s}_by_dataset.png")

    # ---- Part 3: distribution shift quantification ----
    print("\nComputing property panels for shift quantification ...")
    property_names = list(PROPERTIES.keys())
    props_by_split = {
        s: compute_property_matrix(splits[s]["smiles"], args.n_prop_workers)
        for s in splits
    }

    report_lines = [
        "Train / val / test distribution shift report",
        "=" * 72,
        "",
        "Groups: pooled 'cotrain' (all requested datasets combined) plus "
        "each individual source dataset.",
        "domain-gap ratio = mean(NN dist from split to train) / mean(NN "
        "dist within train itself); values near 1 indicate the split sits "
        "on the train manifold, values >> 1 indicate a shift.",
        "",
    ]

    other_splits = {s: splits[s]["embedding"] for s in splits if s != "train"}
    other_2ds = {s: emb2d_by_split[s] for s in splits if s != "train"}
    other_props = {s: props_by_split[s] for s in splits if s != "train"}

    report_lines.extend(quantify_shift_for_group(
        "cotrain (pooled)", splits["train"]["embedding"], other_splits,
        emb2d_train, other_2ds, props_by_split["train"], other_props,
        property_names, args.nn_train_sample, args.nn_eval_sample, args.device, rng,
    ))

    all_sources = sorted(set(splits["train"]["dataset"].tolist()))
    for src in all_sources:
        train_mask = splits["train"]["dataset"] == src
        src_other_embs, src_other_2ds, src_other_props = {}, {}, {}
        for s in other_splits:
            m = splits[s]["dataset"] == src
            src_other_embs[s] = splits[s]["embedding"][m]
            src_other_2ds[s] = emb2d_by_split[s][m]
            src_other_props[s] = props_by_split[s][m]

        report_lines.extend(quantify_shift_for_group(
            src, splits["train"]["embedding"][train_mask], src_other_embs,
            emb2d_train[train_mask], src_other_2ds,
            props_by_split["train"][train_mask], src_other_props,
            property_names, args.nn_train_sample, args.nn_eval_sample, args.device, rng,
        ))

    report_path = args.out_dir / f"{args.prefix}_distribution_shift.txt"
    report_path.write_text("\n".join(report_lines))
    print(f"✓ Saved {report_path}")

    # ---- Part 4: property-colored UMAP figures ----
    prop_datasets = args.property_datasets if args.property_datasets is not None else args.datasets

    prop_smiles: List[str] = []
    prop_emb2d_parts: List[np.ndarray] = []
    prop_dataset_parts: List[np.ndarray] = []

    for s in args.property_splits:
        if s not in splits:
            print(f"[warn] Requested --property-splits includes '{s}' but that "
                  f"split wasn't loaded, skipping.")
            continue
        d = filter_by_datasets(splits[s], prop_datasets) if prop_datasets is not None else splits[s]
        mask = np.isin(splits[s]["dataset"], list(prop_datasets)) if prop_datasets is not None else np.ones(len(splits[s]["smiles"]), dtype=bool)
        prop_smiles.extend(d["smiles"])
        prop_emb2d_parts.append(emb2d_by_split[s][mask])
        prop_dataset_parts.append(splits[s]["dataset"][mask])

    if not prop_smiles:
        print("[warn] No molecules selected for property-colored figures "
              "(check --property-splits / --property-datasets); skipping Part 4.")
    else:
        prop_emb2d = np.concatenate(prop_emb2d_parts, axis=0)

        if args.property_sample_size is not None and len(prop_smiles) > args.property_sample_size:
            sel = rng.choice(len(prop_smiles), size=args.property_sample_size, replace=False)
            prop_smiles = [prop_smiles[i] for i in sel]
            prop_emb2d = prop_emb2d[sel]

        print(f"\nComputing property panel for {len(prop_smiles)} molecules "
              f"(splits={args.property_splits}, datasets={prop_datasets or 'ALL'}) ...")
        prop_matrix = compute_property_matrix(prop_smiles, args.n_prop_workers)

        for p_idx, p_name in enumerate(property_names):
            plot_property_colored(
                prop_emb2d, prop_matrix[:, p_idx], p_name, args.cmap,
                args.out_dir / f"{args.prefix}_property_{p_name}.png",
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
