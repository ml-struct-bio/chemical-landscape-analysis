"""Characterizes and compares the corpus's source datasets, molecular terms only.

Port of the previous pipeline's `9_run_dataset_stats_experiment.py` /
`dataset_stats_analysis.py`. Same algorithm, same defaults, same numbers -- only
the I/O layer changed (plain CSV/NPZ under `data/06_dataset_stats/<tag>/` plus a
manifest, instead of a pickled artifact).

The previous script's spectral half (`--spectral`, live peak re-extraction via a
checkpoint/GPU, reusing script 17's `extract_spectral_corpus`) is NOT ported
here -- script 17 itself has not been ported to this repo yet. This module is
molecular-only, exactly its CPU-only default path.

Two passes, deliberately split by cost:

  - **full-corpus pass** -- canonical SMILES only. Cheap enough (one parse, no
    descriptor evaluation) to run over every molecule, so dataset sizes and
    cross-dataset overlap are exact rather than estimated.
  - **sampled pass** -- the 34-descriptor RDKit panel plus element counts, ring
    topology, stereochemistry and Murcko scaffolds, computed in ONE RDKit parse
    per molecule (`_molecule_worker`). BertzCT, BalabanJ and stereocenter
    enumeration are the slow ones; this runs on a stratified subsample
    (`max_per_dataset`) since every output here is a distribution, and 200k
    molecules resolves one indistinguishably from 2.5M.

Computation only -- never imports matplotlib.
"""
from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from tqdm import tqdm

from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Descriptors import MoleculeDescriptors

from src.analysis.descriptors import DEFAULT_DESCRIPTOR_NAMES
from src.common.manifest import write_manifest
from src.common.workers import safe_n_workers

RDLogger.DisableLog("rdApp.*")

SCHEMA_VERSION = 1

# The subset plotted in the "headline" boxplot/histogram/violin/ECDF grids. The
# full DEFAULT_DESCRIPTOR_NAMES panel (34 wide) still lands in the per-molecule
# CSV and drives the divergence heatmap -- it is just too wide to read as a grid.
HEADLINE_DESCRIPTORS = [
    "MolWt", "MolLogP", "TPSA", "RingCount", "NumRotatableBonds",
    "NumHDonors", "NumHAcceptors", "FractionCSP3", "qed",
]

# Counted per molecule; anything outside this list folds into "other".
TRACKED_ELEMENTS = ["N", "O", "S", "P", "F", "Cl", "Br", "I"]

RING_FEATURES = [
    "n_rings", "n_aromatic_rings", "n_aliphatic_rings", "max_ring_size",
    "n_macrocycles", "n_spiro_atoms", "n_bridgehead_atoms",
]
STEREO_FEATURES = ["n_stereocenters", "n_unspecified_stereocenters", "frac_stereo_specified"]
COMPOSITION_FEATURES = [f"n_{el}" for el in TRACKED_ELEMENTS] + [
    "n_heavy_atoms", "n_halogens", "frac_carbon",
]
ALL_NUMERIC_FEATURES = list(DEFAULT_DESCRIPTOR_NAMES) + COMPOSITION_FEATURES + RING_FEATURES + STEREO_FEATURES

# A ring this size or larger counts as a macrocycle.
MACROCYCLE_MIN_RING_SIZE = 12


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------

def load_dataset_stats_data(data_dir: Path, prefix: str, splits: Sequence[str]) -> pd.DataFrame:
    import torch

    frames: List[pd.DataFrame] = []
    for split in splits:
        path = Path(data_dir) / f"{prefix}_{split}_global_cond.pt"
        if not path.exists():
            raise SystemExit(f"Expected extraction output not found: {path}\n"
                             f"Run extraction/00_global_cond.py first.")
        d = torch.load(path, map_location="cpu", weights_only=False)
        smiles = list(d["smiles"])
        datasets = list(map(str, d["dataset"]))
        if len(smiles) != len(datasets):
            raise ValueError(f"Length mismatch in {path}: {len(smiles)} smiles vs "
                              f"{len(datasets)} dataset labels")
        frames.append(pd.DataFrame({"split": split, "dataset": datasets, "smiles": smiles}))
    return pd.concat(frames, ignore_index=True)


def stratified_sample(df: pd.DataFrame, max_per_dataset: Optional[int], seed: int) -> pd.DataFrame:
    """Caps each dataset at `max_per_dataset` rows. `None`/0 means no cap."""
    if not max_per_dataset:
        return df
    rng = np.random.default_rng(seed)
    keep: List[np.ndarray] = []
    for _, group in df.groupby("dataset", sort=True):
        idx = group.index.to_numpy()
        if len(idx) > max_per_dataset:
            idx = rng.choice(idx, max_per_dataset, replace=False)
        keep.append(idx)
    return df.loc[np.sort(np.concatenate(keep))].reset_index(drop=True)


# -----------------------------------------------------------------------------
# Pass 1 (full corpus, cheap): canonical SMILES -> exact overlap
# -----------------------------------------------------------------------------

def _canonical_worker(smi: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol is not None else None


def compute_canonical_smiles(smiles_list: Sequence[str], n_workers: int) -> List[Optional[str]]:
    if n_workers <= 1:
        return [_canonical_worker(s) for s in tqdm(smiles_list, desc="Canonical SMILES")]
    with mp.Pool(n_workers) as pool:
        return list(tqdm(pool.imap(_canonical_worker, smiles_list, chunksize=512),
                          total=len(smiles_list), desc=f"Canonical SMILES ({n_workers} workers)"))


def compute_overlap_tables(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Exact cross-dataset molecule sharing, on canonical SMILES.

    Returns (pairwise, per_dataset). `pairwise` has one row per unordered
    dataset pair with the intersection size and Jaccard index; `per_dataset`
    reports each dataset's unique-molecule count, internal duplicate rate, and
    how much of it is shared with any other dataset. Relevant here because
    whether the training mixture was deduplicated across sources is exactly
    what the `dedupON`/`dedupOFF` checkpoint families differ on."""
    valid = df[df["canonical_smiles"].notna()]
    by_dataset = {ds: set(g["canonical_smiles"]) for ds, g in valid.groupby("dataset", sort=True)}
    names = sorted(by_dataset)

    pair_rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            inter = len(by_dataset[a] & by_dataset[b])
            union = len(by_dataset[a] | by_dataset[b])
            pair_rows.append({
                "dataset_a": a, "dataset_b": b, "n_shared": inter,
                "jaccard": float(inter / union) if union else np.nan,
                "frac_of_a": float(inter / len(by_dataset[a])) if by_dataset[a] else np.nan,
                "frac_of_b": float(inter / len(by_dataset[b])) if by_dataset[b] else np.nan,
            })

    per_rows = []
    for ds in names:
        rows = valid[valid["dataset"] == ds]
        others: set = set()
        for other in names:
            if other != ds:
                others |= by_dataset[other]
        n_unique = len(by_dataset[ds])
        per_rows.append({
            "dataset": ds,
            "n_molecules": int(len(rows)),
            "n_unique": int(n_unique),
            "internal_duplicate_frac": float(1 - n_unique / len(rows)) if len(rows) else np.nan,
            "n_shared_with_others": int(len(by_dataset[ds] & others)),
            "frac_shared_with_others": float(len(by_dataset[ds] & others) / n_unique) if n_unique else np.nan,
        })
    return pd.DataFrame(pair_rows), pd.DataFrame(per_rows)


# -----------------------------------------------------------------------------
# Pass 2 (sampled): descriptors + composition + rings + stereo + scaffold
# -----------------------------------------------------------------------------

_CALC = None


def _init_worker() -> None:
    global _CALC
    _CALC = MoleculeDescriptors.MolecularDescriptorCalculator(list(DEFAULT_DESCRIPTOR_NAMES))


def _molecule_worker(smi: str) -> Dict[str, Any]:
    """Everything derivable from one RDKit parse, computed in one place so the
    molecule is built exactly once instead of once per analysis."""
    nan_row: Dict[str, Any] = {name: np.nan for name in DEFAULT_DESCRIPTOR_NAMES}
    nan_row.update({name: np.nan for name in COMPOSITION_FEATURES + RING_FEATURES + STEREO_FEATURES})
    nan_row["murcko_scaffold"] = None
    nan_row["valid"] = False

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return nan_row

    row: Dict[str, Any] = {"valid": True}
    try:
        for name, val in zip(_CALC.descriptorNames, _CALC.CalcDescriptors(mol)):
            row[name] = np.nan if (val is None or not np.isfinite(val)) else float(val)
    except Exception:
        return nan_row

    # --- composition ---
    counts = {el: 0 for el in TRACKED_ELEMENTS}
    n_carbon = 0
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym == "C":
            n_carbon += 1
        elif sym in counts:
            counts[sym] += 1
    n_heavy = mol.GetNumHeavyAtoms()
    for el in TRACKED_ELEMENTS:
        row[f"n_{el}"] = float(counts[el])
    row["n_heavy_atoms"] = float(n_heavy)
    row["n_halogens"] = float(sum(counts[el] for el in ("F", "Cl", "Br", "I")))
    row["frac_carbon"] = float(n_carbon / n_heavy) if n_heavy else np.nan

    # --- rings ---
    ring_info = mol.GetRingInfo()
    ring_sizes = [len(r) for r in ring_info.AtomRings()]
    row["n_rings"] = float(len(ring_sizes))
    row["n_aromatic_rings"] = float(rdMolDescriptors.CalcNumAromaticRings(mol))
    row["n_aliphatic_rings"] = float(rdMolDescriptors.CalcNumAliphaticRings(mol))
    row["max_ring_size"] = float(max(ring_sizes)) if ring_sizes else 0.0
    row["n_macrocycles"] = float(sum(1 for s in ring_sizes if s >= MACROCYCLE_MIN_RING_SIZE))
    row["n_spiro_atoms"] = float(rdMolDescriptors.CalcNumSpiroAtoms(mol))
    row["n_bridgehead_atoms"] = float(rdMolDescriptors.CalcNumBridgeheadAtoms(mol))

    # --- stereochemistry ---
    # Uses the "potential" stereocenter count so an undefined center still
    # counts toward the denominator; that ratio is what separates a curated
    # natural-product source from a reaction corpus with stereo stripped.
    try:
        n_total = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
        n_unspec = rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(mol)
    except Exception:
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True, flagPossibleStereoCenters=True)
        n_total = sum(1 for a in mol.GetAtoms() if a.HasProp("_ChiralityPossible") or a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED)
        n_unspec = sum(1 for a in mol.GetAtoms() if a.HasProp("_ChiralityPossible") and a.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED)
    row["n_stereocenters"] = float(n_total)
    row["n_unspecified_stereocenters"] = float(n_unspec)
    row["frac_stereo_specified"] = float((n_total - n_unspec) / n_total) if n_total else np.nan

    # --- scaffold ---
    try:
        row["murcko_scaffold"] = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        row["murcko_scaffold"] = None
    return row


def compute_molecule_table(smiles_list: Sequence[str], n_workers: int) -> pd.DataFrame:
    """One RDKit pass producing every per-molecule feature at once."""
    if n_workers <= 1:
        _init_worker()
        rows = [_molecule_worker(s) for s in tqdm(smiles_list, desc="Molecular features")]
    else:
        with mp.Pool(n_workers, initializer=_init_worker) as pool:
            rows = list(tqdm(pool.imap(_molecule_worker, smiles_list, chunksize=256),
                              total=len(smiles_list),
                              desc=f"Molecular features ({n_workers} workers)"))
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Summaries
# -----------------------------------------------------------------------------

def summarize_by_dataset(stats_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset_name, group in stats_df.groupby("dataset", sort=True):
        row: Dict[str, Any] = {
            "dataset": dataset_name,
            "n_molecules": int(len(group)),
            "n_valid_smiles": int(group["valid"].sum()),
            "valid_smiles_fraction": float(group["valid"].mean()),
        }
        for feat in ALL_NUMERIC_FEATURES:
            if feat not in group:
                continue
            vals = group[feat].dropna()
            row[f"mean_{feat}"] = float(vals.mean()) if len(vals) else np.nan
            row[f"std_{feat}"] = float(vals.std()) if len(vals) else np.nan
            row[f"median_{feat}"] = float(vals.median()) if len(vals) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def compute_pairwise_divergence(stats_df: pd.DataFrame, features: Sequence[str],
                                 max_ks_sample: int = 20000, seed: int = 1234) -> pd.DataFrame:
    """Per-feature KS statistic + standardized mean difference for every
    unordered dataset pair -- the "how do these sources actually differ" table.

    KS is capped at `max_ks_sample` points per side: the statistic converges
    quickly, while `ks_2samp` on two 200k-point vectors for 34 features x 3
    pairs does not."""
    rng = np.random.default_rng(seed)
    datasets = sorted(stats_df["dataset"].unique())
    rows = []
    for i, a in enumerate(datasets):
        for b in datasets[i + 1:]:
            for feat in features:
                if feat not in stats_df:
                    continue
                va = stats_df.loc[stats_df["dataset"] == a, feat].dropna().to_numpy()
                vb = stats_df.loc[stats_df["dataset"] == b, feat].dropna().to_numpy()
                if len(va) < 10 or len(vb) < 10:
                    continue
                sa = rng.choice(va, max_ks_sample, replace=False) if len(va) > max_ks_sample else va
                sb = rng.choice(vb, max_ks_sample, replace=False) if len(vb) > max_ks_sample else vb
                stat, pval = ks_2samp(sa, sb)
                pooled = np.sqrt((va.var(ddof=1) + vb.var(ddof=1)) / 2.0)
                rows.append({
                    "dataset_a": a, "dataset_b": b, "feature": feat,
                    "mean_a": float(va.mean()), "mean_b": float(vb.mean()),
                    "standardized_diff": float((va.mean() - vb.mean()) / (pooled + 1e-12)),
                    "ks_stat": float(stat), "ks_pvalue": float(pval),
                })
    return pd.DataFrame(rows)


def compute_scaffold_stats(stats_df: pd.DataFrame, top_k: int = 15
                            ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(summary, top_scaffolds).

    `summary` carries each dataset's scaffold count, scaffolds-per-molecule
    ratio, and the fraction of molecules covered by its top 1%/10% of
    scaffolds -- a concentration measure, so a source built from a few
    decorated cores separates from a structurally diverse one."""
    summary_rows, top_rows = [], []
    for ds, group in stats_df.groupby("dataset", sort=True):
        scaf = group["murcko_scaffold"].dropna()
        scaf = scaf[scaf != ""]
        if scaf.empty:
            continue
        counts = scaf.value_counts()
        n_mol, n_scaf = len(scaf), len(counts)
        cum = counts.cumsum() / n_mol
        summary_rows.append({
            "dataset": ds,
            "n_molecules_with_scaffold": int(n_mol),
            "n_unique_scaffolds": int(n_scaf),
            "scaffolds_per_molecule": float(n_scaf / n_mol),
            "frac_acyclic": float((scaf == "").sum() / n_mol),
            "top1pct_scaffold_coverage": float(cum.iloc[max(0, int(np.ceil(0.01 * n_scaf)) - 1)]),
            "top10pct_scaffold_coverage": float(cum.iloc[max(0, int(np.ceil(0.10 * n_scaf)) - 1)]),
        })
        for rank, (smi, n) in enumerate(counts.head(top_k).items(), start=1):
            top_rows.append({"dataset": ds, "rank": rank, "scaffold": smi,
                              "n_molecules": int(n), "frac": float(n / n_mol)})
    return pd.DataFrame(summary_rows), pd.DataFrame(top_rows)


def scaffold_coverage_curves(stats_df: pd.DataFrame, n_points: int = 200) -> pd.DataFrame:
    """Long-form (dataset, frac_scaffolds, frac_molecules) table, scaffolds
    ranked most-common first -- a plain CSV rather than a dict of arrays, since
    figure data here must stay tabular."""
    rows = []
    for ds, group in stats_df.groupby("dataset", sort=True):
        scaf = group["murcko_scaffold"].dropna()
        if scaf.empty:
            continue
        counts = scaf.value_counts().to_numpy()
        cum = np.cumsum(counts) / counts.sum()
        x = np.linspace(0, 1, min(n_points, len(cum)))
        idx = np.clip((x * (len(cum) - 1)).astype(int), 0, len(cum) - 1)
        for xi, yi in zip(x, cum[idx]):
            rows.append({"dataset": ds, "frac_scaffolds": float(xi), "frac_molecules": float(yi)})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------------

def run_dataset_stats(
    *,
    data_dir: Path,
    prefix: str,
    splits: Sequence[str],
    out_dir: Path,
    tag: str,
    slug: str,
    datasets: Optional[Sequence[str]] = None,
    n_workers: int = safe_n_workers(),
    max_per_dataset: Optional[int] = 200_000,
    top_k_scaffolds: int = 15,
    seed: int = 1234,
) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}

    df = load_dataset_stats_data(data_dir, prefix, splits)
    if datasets is not None:
        df = df[df["dataset"].isin(list(datasets))].copy().reset_index(drop=True)
    if df.empty:
        raise SystemExit("No molecules were loaded from the requested data files.")

    counts_df = (df.groupby(["dataset", "split"], sort=True).size()
                 .rename("n_molecules").reset_index())
    print(f"Loaded {len(df)} molecules across {df['dataset'].nunique()} datasets "
          f"and {df['split'].nunique()} split(s).")

    # --- pass 1: full corpus, canonical SMILES only -> exact overlap ---
    print("\n### Pass 1/2: canonical SMILES over the FULL corpus (exact overlap) ###")
    df["canonical_smiles"] = compute_canonical_smiles(df["smiles"].tolist(), n_workers)
    overlap_pairs_df, overlap_summary_df = compute_overlap_tables(df)

    # --- pass 2: sampled, everything expensive ---
    sampled = stratified_sample(df, max_per_dataset, seed)
    if len(sampled) < len(df):
        print(f"\n### Pass 2/2: molecular features on a stratified subsample "
              f"({len(sampled)} of {len(df)} molecules, cap {max_per_dataset}/dataset) ###")
    else:
        print(f"\n### Pass 2/2: molecular features on all {len(sampled)} molecules ###")
    feature_df = compute_molecule_table(sampled["smiles"].tolist(), n_workers)
    stats_df = pd.concat([sampled.reset_index(drop=True), feature_df], axis=1)

    summary_df = summarize_by_dataset(stats_df)
    divergence_df = compute_pairwise_divergence(stats_df, ALL_NUMERIC_FEATURES, seed=seed)
    scaffold_summary_df, scaffold_top_df = compute_scaffold_stats(stats_df, top_k=top_k_scaffolds)
    coverage_df = scaffold_coverage_curves(stats_df)

    tables = {
        "stats": stats_df,
        "counts": counts_df,
        "summary": summary_df,
        "divergence": divergence_df,
        "overlap_pairs": overlap_pairs_df,
        "overlap_summary": overlap_summary_df,
        "scaffold_summary": scaffold_summary_df,
        "scaffold_top": scaffold_top_df,
        "scaffold_coverage": coverage_df,
    }
    for name, table in tables.items():
        path = out_dir / f"{name}.csv"
        table.to_csv(path, index=False)
        written[name] = path
        print(f"Saved {path}")

    write_manifest(
        out_dir, slug=slug, tag=tag, schema_version=SCHEMA_VERSION,
        params={
            "prefix": prefix, "splits": list(splits), "data_dir": str(data_dir),
            "datasets": sorted(df["dataset"].unique().tolist()),
            "n_total_molecules": int(len(df)),
            "n_sampled_molecules": int(len(sampled)),
            "max_per_dataset": max_per_dataset,
            "top_k_scaffolds": top_k_scaffolds,
            "seed": seed,
            "headline_descriptors": list(HEADLINE_DESCRIPTORS),
            "composition_features": list(COMPOSITION_FEATURES),
            "ring_features": list(RING_FEATURES),
            "stereo_features": list(STEREO_FEATURES),
            "all_numeric_features": list(ALL_NUMERIC_FEATURES),
        },
        inputs=[Path(data_dir) / f"{prefix}_{s}_global_cond.pt" for s in splits],
        outputs=list(written.values()),
    )
    return written
