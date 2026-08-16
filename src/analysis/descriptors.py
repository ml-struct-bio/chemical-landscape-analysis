"""RDKit descriptor panel, and PC-vs-anything Pearson correlation tables.

Carried over from the previous pipeline's `pca_traversal_analysis.py`, minus the
two functions that had no business being there: `mol_to_image` (a renderer, now
`src/plotting/mol_render.py`) and `nearest_molecule` (a search, now
`src/analysis/geometry.py`). Every plotting module in the old repo imported
those two *from the analysis layer*, which is how the layer separation eroded.

The correlation helpers are deliberately generic over the second matrix: the
same code correlates PCs against RDKit descriptors and against the NMR spectral
feature panel, which is what lets one PCA carry two explanations.
"""
from __future__ import annotations

import multiprocessing as mp
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from tqdm import tqdm

from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors


DEFAULT_DESCRIPTOR_NAMES = [
    "MaxAbsEStateIndex", "MaxEStateIndex", "MinAbsEStateIndex", "MinEStateIndex",
    "qed", "SPS", "MolWt", "HeavyAtomMolWt", "ExactMolWt", "NumValenceElectrons",
    "NumRadicalElectrons", "BalabanJ", "BertzCT", "HallKierAlpha", "LabuteASA",
    "TPSA", "FractionCSP3", "HeavyAtomCount", "NHOHCount", "NOCount",
    "NumAliphaticCarbocycles", "NumAliphaticHeterocycles", "NumAliphaticRings",
    "NumAromaticCarbocycles", "NumAromaticHeterocycles", "NumAromaticRings",
    "NumHAcceptors", "NumHDonors", "NumHeteroatoms", "NumRotatableBonds",
    "NumSaturatedCarbocycles", "NumSaturatedHeterocycles", "NumSaturatedRings",
    "RingCount", "MolLogP",
]


_CALC = None  # set once per worker process


def _init_descriptor_worker(descriptor_names: List[str]) -> None:
    global _CALC
    _CALC = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)


def _descriptor_worker(smi: str) -> List[float]:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return [np.nan] * len(_CALC.descriptorNames)
    vals = _CALC.CalcDescriptors(mol)
    return [np.nan if (v is None or not np.isfinite(v)) else float(v) for v in vals]


def compute_descriptor_matrix(smiles_list: Sequence[str], descriptor_names: List[str],
                              n_workers: int = 1) -> np.ndarray:
    """(n_molecules, n_descriptors) float64, NaN for unparseable SMILES.

    The dominant cost of a full-corpus run -- hence the worker pool, and hence
    the on-disk cache in `src/analysis/corpus.py` that lets many embeddings over
    the same molecule set reuse one pass.
    """
    if n_workers <= 1:
        _init_descriptor_worker(descriptor_names)
        rows = [_descriptor_worker(s) for s in tqdm(smiles_list, desc="RDKit descriptors")]
    else:
        with mp.Pool(n_workers, initializer=_init_descriptor_worker,
                     initargs=(descriptor_names,)) as pool:
            rows = list(
                tqdm(pool.imap(_descriptor_worker, smiles_list, chunksize=256),
                     total=len(smiles_list),
                     desc=f"RDKit descriptors ({n_workers} workers)")
            )
    return np.array(rows, dtype=np.float64)


def compute_pc_correlations(pcs: np.ndarray, value_matrix: np.ndarray,
                            value_names: List[str], desc: str = "PC correlations") -> pd.DataFrame:
    """Pearson r between every PC and every column of `value_matrix`.

    Rows are `PC1..PCn`, columns are `value_names`. Pairs are masked to rows
    where both sides are finite, and a column with fewer than 10 usable rows is
    left NaN rather than reported from near-nothing.
    """
    n_pcs = pcs.shape[1]
    corr = np.full((n_pcs, len(value_names)), np.nan)

    for pc_idx in tqdm(range(n_pcs), desc=desc):
        pc_values = pcs[:, pc_idx]
        for v_idx, _ in enumerate(value_names):
            values = value_matrix[:, v_idx]
            mask = np.isfinite(values) & np.isfinite(pc_values)
            if mask.sum() < 10:
                continue
            r, _ = pearsonr(pc_values[mask], values[mask])
            corr[pc_idx, v_idx] = r

    return pd.DataFrame(corr, index=[f"PC{i+1}" for i in range(n_pcs)], columns=value_names)


def best_per_pc(corr_df: pd.DataFrame, column: str = "descriptor") -> pd.DataFrame:
    """Each PC's single most strongly correlated column, ranked by |r| with the
    sign preserved in the reported `r`."""
    rows = []
    for pc in corr_df.index:
        row = corr_df.loc[pc]
        abs_row = row.abs()
        if abs_row.isna().all():
            rows.append({"pc": pc, column: None, "r": np.nan})
            continue
        best = abs_row.idxmax()
        rows.append({"pc": pc, column: best, "r": row[best]})
    return pd.DataFrame(rows)


def top_per_pc(corr_df: pd.DataFrame, top_k: int = 5, column: str = "feature") -> pd.DataFrame:
    """The `top_k` most strongly correlated columns per PC, long-form."""
    rows = []
    for pc in corr_df.index:
        row = corr_df.loc[pc]
        ranked = row.abs().sort_values(ascending=False)
        for rank, name in enumerate(ranked.index[:top_k], start=1):
            r = row[name]
            if not np.isfinite(r):
                continue
            rows.append({"pc": pc, "rank": rank, column: name, "r": float(r),
                         "abs_r": float(abs(r))})
    return pd.DataFrame(rows)


def correlation_report(corr_df: pd.DataFrame, what: str) -> str:
    """The human-readable dump that used to be written as a `.txt` beside the
    CSV. It is analysis output, not a figure -- the numbers are already decided
    by the time anything draws."""
    return (f"Raw Pearson correlation (r) between each PC and each {what}.\n"
            f"Rows = principal components, columns = {what}s.\n\n"
            + corr_df.round(4).to_string() + "\n")
