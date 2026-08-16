"""RDKit molecular property panels for colouring maps.

Carried over from the previous pipeline's `umap_analysis.py`. Distinct from
`src/analysis/descriptors.py`, which computes a fixed panel for PC correlations:
this one is user-selectable (`basic` / `extended` / `all` / explicit names) and
is what the UMAP colourings draw from.
"""
from __future__ import annotations

import multiprocessing as mp
from typing import Dict, List, Optional, Sequence

import numpy as np
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import Descriptors


def _aromatic_fraction(mol) -> float:
    atoms = mol.GetAtoms()
    if not mol.GetNumAtoms():
        return 0.0
    return sum(1 for atom in atoms if atom.GetIsAromatic()) / mol.GetNumAtoms()


# The five properties every documented run of the previous script produced.
BASIC_PROPERTY_NAMES = ["MolWt", "LogP", "TPSA", "RingCount", "Aromaticity"]

# A wider panel spanning distinct chemical axes rather than 200 mutually
# redundant descriptors: size, lipophilicity, polarity/H-bonding, flexibility,
# ring systems, topological complexity, charge, and drug-likeness.
EXTENDED_PROPERTY_NAMES = [
    # size / bulk
    "MolWt", "HeavyAtomCount", "NumValenceElectrons", "LabuteASA",
    # lipophilicity
    "LogP", "MolMR",
    # polarity + H-bonding
    "TPSA", "NumHDonors", "NumHAcceptors", "NHOHCount", "NOCount",
    "NumHeteroatoms", "MaxPartialCharge", "MinPartialCharge",
    # saturation / flexibility
    "FractionCSP3", "NumRotatableBonds", "HallKierAlpha",
    # ring systems
    "RingCount", "NumAromaticRings", "NumAliphaticRings", "NumSaturatedRings",
    "NumAromaticHeterocycles", "NumAromaticCarbocycles", "NumSaturatedHeterocycles",
    "Aromaticity",
    # topological complexity / shape indices
    "BalabanJ", "BertzCT", "Kappa1", "Kappa2", "Kappa3", "Chi0v", "Chi1v", "Chi2v",
    # composite
    "qed",
]

# Excluded from `--properties all`. Ipc grows factorially with molecule size and
# overflows to inf on ordinary drug-sized inputs, which saturates any colormap it
# lands in.
ALL_EXCLUDED = {"Ipc"}

_RDKIT_DESCRIPTORS = dict(Descriptors.descList)
PROPERTY_CATALOG = {
    **_RDKIT_DESCRIPTORS,
    "LogP": Descriptors.MolLogP,     # alias kept so old runs reproduce
    "Aromaticity": _aromatic_fraction,
}

# The live panel. `_property_worker` reads this at call time and the pool is
# forked after `set_property_funcs`, so workers inherit whatever is set here.
PROPERTY_FUNCS: Dict[str, object] = {n: PROPERTY_CATALOG[n] for n in BASIC_PROPERTY_NAMES}


def resolve_property_names(spec: Optional[Sequence[str]]) -> List[str]:
    """Turn a `--properties` spec into an ordered, de-duplicated name list."""
    if not spec:
        return list(BASIC_PROPERTY_NAMES)

    names: List[str] = []
    unknown: List[str] = []
    for item in spec:
        if item == "basic":
            names.extend(BASIC_PROPERTY_NAMES)
        elif item == "extended":
            names.extend(EXTENDED_PROPERTY_NAMES)
        elif item == "all":
            names.extend(n for n in _RDKIT_DESCRIPTORS if n not in ALL_EXCLUDED)
            names.append("Aromaticity")
        elif item in PROPERTY_CATALOG:
            names.append(item)
        else:
            unknown.append(item)

    if unknown:
        raise SystemExit(
            f"Unknown propert{'y' if len(unknown) == 1 else 'ies'}: {unknown}.\n"
            f"Use 'basic', 'extended', 'all', or any of the {len(PROPERTY_CATALOG)} "
            f"names in properties.PROPERTY_CATALOG (RDKit's Descriptors.descList plus "
            f"'LogP' and 'Aromaticity').")

    seen = set()
    return [n for n in names if not (n in seen or seen.add(n))]


def set_property_funcs(names: Sequence[str]) -> List[str]:
    """Point the live panel at `names`. Call before any worker pool starts."""
    resolved = list(names)
    PROPERTY_FUNCS.clear()
    PROPERTY_FUNCS.update({n: PROPERTY_CATALOG[n] for n in resolved})
    return resolved


def _property_worker(smi: str) -> List[float]:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return [np.nan] * len(PROPERTY_FUNCS)
    out = []
    for fn in PROPERTY_FUNCS.values():
        try:
            val = fn(mol)
            out.append(float(val) if val is not None and np.isfinite(val) else np.nan)
        except Exception:
            out.append(np.nan)
    return out


def compute_properties(smiles: Sequence[str], names: Sequence[str],
                       n_workers: int = 1) -> np.ndarray:
    """(n_molecules, n_properties) float32, NaN where RDKit failed."""
    set_property_funcs(names)
    if n_workers <= 1:
        rows = [_property_worker(s) for s in tqdm(smiles, desc="RDKit properties")]
    else:
        with mp.Pool(n_workers) as pool:
            rows = list(tqdm(pool.imap(_property_worker, smiles, chunksize=256),
                             total=len(smiles),
                             desc=f"RDKit properties ({n_workers} workers)"))
    return np.asarray(rows, dtype=np.float32)
