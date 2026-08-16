"""The spectral feature panel computed off a datamodule's peak dict.

Lifted verbatim from the previous pipeline's
`src/analysis/real_vs_synthetic_analysis.py` (lines 290-503): the feature-name
lists, the per-axis / multiplet / nH statistics, the spectrum-vs-structure
quality features and their structure controls, and
`compute_spectral_feature_matrix` which assembles them.

Nothing here touches a model or a GPU -- the features come off the peak lists
the model is conditioned on, which is why `extraction/01_spectral_features.py`
runs CPU-only.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import kurtosis, skew

try:
    from rdkit import Chem
except Exception:  # pragma: no cover - optional dependency
    Chem = None


SPECTRAL_FEATURE_NAMES = [
    "n_H_peaks", "H_shift_min", "H_shift_max", "H_shift_range", "H_shift_mean",
    "H_shift_median", "H_shift_std", "H_shift_skew", "H_shift_kurtosis",
    "H_mean_peak_spacing", "H_min_peak_spacing", "H_peaks_per_ppm",
    "n_C_peaks", "C_shift_min", "C_shift_max", "C_shift_range", "C_shift_mean",
    "C_shift_median", "C_shift_std", "C_shift_skew", "C_shift_kurtosis",
    "C_mean_peak_spacing", "C_min_peak_spacing", "C_peaks_per_ppm",
    "H_to_C_peak_ratio", "total_peaks",
]


def _one_axis_features(prefix: str, values: np.ndarray) -> Dict[str, float]:
    n = len(values)
    out = {f"{prefix}_shift_min": np.nan, f"{prefix}_shift_max": np.nan, f"{prefix}_shift_range": np.nan, f"{prefix}_shift_mean": np.nan, f"{prefix}_shift_median": np.nan, f"{prefix}_shift_std": np.nan, f"{prefix}_shift_skew": np.nan, f"{prefix}_shift_kurtosis": np.nan, f"{prefix}_mean_peak_spacing": np.nan, f"{prefix}_min_peak_spacing": np.nan, f"{prefix}_peaks_per_ppm": np.nan}
    out[f"n_{prefix}_peaks"] = n
    if n == 0:
        return out
    v = np.sort(values)
    shift_range = float(v[-1] - v[0])
    out[f"{prefix}_shift_min"] = float(v[0])
    out[f"{prefix}_shift_max"] = float(v[-1])
    out[f"{prefix}_shift_range"] = shift_range
    out[f"{prefix}_shift_mean"] = float(v.mean())
    out[f"{prefix}_shift_median"] = float(np.median(v))
    if n > 1:
        out[f"{prefix}_shift_std"] = float(v.std(ddof=1))
        gaps = np.diff(v)
        out[f"{prefix}_mean_peak_spacing"] = float(gaps.mean())
        out[f"{prefix}_min_peak_spacing"] = float(gaps.min())
        if shift_range > 0:
            out[f"{prefix}_peaks_per_ppm"] = n / shift_range
    if n > 2:
        out[f"{prefix}_shift_skew"] = float(skew(v))
        out[f"{prefix}_shift_kurtosis"] = float(kurtosis(v))
    return out


def _multiplet_features(values: np.ndarray) -> Dict[str, float]:
    n = len(values)
    out = {"H_multiplet_n_unique": np.nan, "H_multiplet_mode_frac": np.nan, "H_multiplet_entropy": np.nan}
    if n == 0:
        return out
    _, counts = np.unique(values, return_counts=True)
    probs = counts / n
    out["H_multiplet_n_unique"] = float(len(counts))
    out["H_multiplet_mode_frac"] = float(counts.max() / n)
    out["H_multiplet_entropy"] = float(-(probs * np.log(probs + 1e-12)).sum())
    return out


def _nh_features(values: np.ndarray) -> Dict[str, float]:
    n = len(values)
    out = {"H_nH_mean": np.nan, "H_nH_max": np.nan, "H_nH_sum": np.nan, "H_frac_CH": np.nan, "H_frac_CH2": np.nan, "H_frac_CH3": np.nan}
    if n == 0:
        return out
    out["H_nH_mean"] = float(values.mean())
    out["H_nH_max"] = float(values.max())
    out["H_nH_sum"] = float(values.sum())
    out["H_frac_CH"] = float(np.mean(values == 1))
    out["H_frac_CH2"] = float(np.mean(values == 2))
    out["H_frac_CH3"] = float(np.mean(values == 3))
    return out


QUALITY_FEATURE_NAMES = ["H_proton_balance", "H_proton_balance_frac", "C_peak_completeness"]

# Raw structure counts, carried as controls. They are the references the
# quality features are measured against, so having them in the panel is what
# lets you tell a real completeness signal from the molecular-size signal that
# every count feature already carries. Unlike the quality features these say
# nothing about the spectrum, so a truncated peak list does NOT invalidate
# them -- they are NaN only for an unparseable SMILES.
STRUCTURE_FEATURE_NAMES = ["n_H_formula", "n_C_formula"]

# Which protons the spectrum is expected to account for. Measured on the
# cotrain-v3 corpus (60k molecules): against TOTAL hydrogens the 1H integration
# sum is exact for 93.6% of nmrexp-v3 and 56.6% of uspto-v3 with median
# difference 0, while against carbon-bound hydrogens only it is exact for
# 69.7%/34.7% -- i.e. these peak lists DO include exchangeable OH/NH protons,
# so total H is the correct reference. "carbon_bound" is kept for datasets
# whose peak lists omit exchangeables.
DEFAULT_H_REFERENCE = "total"


def _structure_reference(smi: Optional[str], h_reference: str = DEFAULT_H_REFERENCE) -> Tuple[float, float, float]:
    """(expected proton count, symmetry-unique carbon count, total carbon count).

    The middle value counts symmetry-DISTINCT carbons, not carbons: equivalent
    carbons give one 13C signal, so comparing peak count to raw carbon count
    would score symmetric molecules as defective spectra. Canonical ranking
    with `breakTies=False` yields the equivalence classes. The raw carbon
    count is returned alongside it as the control; their ratio is the
    molecule's carbon-symmetry factor.
    """
    mol = Chem.MolFromSmiles(smi) if smi else None
    if mol is None:
        return np.nan, np.nan, np.nan
    mol_h = Chem.AddHs(mol)
    if h_reference == "carbon_bound":
        n_h = sum(1 for a in mol_h.GetAtoms()
                  if a.GetAtomicNum() == 1
                  and a.GetNeighbors() and a.GetNeighbors()[0].GetAtomicNum() == 6)
    else:
        n_h = sum(1 for a in mol_h.GetAtoms() if a.GetAtomicNum() == 1)
    carbons = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 6]
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    n_c_unique = len({ranks[a.GetIdx()] for a in carbons})
    return float(n_h), float(n_c_unique), float(len(carbons))


def _quality_features(nh_sum: float, n_c_peaks: float, n_h_ref: float, n_c_unique: float,
                       h_truncated: bool, c_truncated: bool) -> Dict[str, float]:
    """Spectrum-vs-structure consistency. NaN wherever the comparison would be
    meaningless: unparseable SMILES, no protons/carbons to divide by, or a peak
    list that hit the padding bound (a truncated list looks deficient no matter
    how good the spectrum is, so those molecules are dropped from these
    features rather than allowed to bias their correlations)."""
    out = {name: np.nan for name in QUALITY_FEATURE_NAMES}
    if not h_truncated and np.isfinite(n_h_ref) and np.isfinite(nh_sum):
        balance = nh_sum - n_h_ref
        out["H_proton_balance"] = float(balance)
        if n_h_ref > 0:
            out["H_proton_balance_frac"] = float(balance / n_h_ref)
    if not c_truncated and np.isfinite(n_c_unique) and n_c_unique > 0 and n_c_peaks > 0:
        out["C_peak_completeness"] = float(n_c_peaks / n_c_unique)
    return out


def compute_spectral_feature_matrix(condition: Dict[str, np.ndarray], idx: np.ndarray, h_shift_key: str, c_shift_key: str, h_mask_key: Optional[str], c_mask_key: Optional[str], h_nh_key: Optional[str] = None, h_multiplet_key: Optional[str] = None, smiles: Optional[Sequence[str]] = None, h_reference: str = DEFAULT_H_REFERENCE, include_structure_controls: bool = True) -> Tuple[Optional[np.ndarray], List[str]]:
    missing = [k for k in (h_shift_key, c_shift_key) if k not in condition]
    if missing:
        print(f"[warn] Spectral feature keys not found in condition dict: {missing}. Skipping spectral correlation for this pair.")
        return None, []

    h_shifts_all = condition[h_shift_key][idx]
    c_shifts_all = condition[c_shift_key][idx]
    h_mask_all = condition[h_mask_key][idx] if (h_mask_key and h_mask_key in condition) else None
    c_mask_all = condition[c_mask_key][idx] if (c_mask_key and c_mask_key in condition) else None
    h_nh_all = condition[h_nh_key][idx] if (h_nh_key and h_nh_key in condition) else None
    h_mult_all = condition[h_multiplet_key][idx] if (h_multiplet_key and h_multiplet_key in condition) else None

    feature_names = list(SPECTRAL_FEATURE_NAMES)
    if h_nh_all is not None:
        feature_names += ["H_nH_mean", "H_nH_max", "H_nH_sum", "H_frac_CH", "H_frac_CH2", "H_frac_CH3"]
    if h_mult_all is not None:
        feature_names += ["H_multiplet_n_unique", "H_multiplet_mode_frac", "H_multiplet_entropy"]
    # The quality features compare the spectrum against the STRUCTURE, so they
    # are the only ones needing SMILES; without them the panel is unchanged.
    want_quality = smiles is not None and h_nh_all is not None
    if smiles is not None and h_nh_all is None:
        print(f"[warn] SMILES given but no '{h_nh_key}' in the condition dict -- the 1H "
              "integration sum is unavailable, so the spectrum-vs-structure quality "
              "features are skipped.")
    want_controls = want_quality and include_structure_controls
    if want_quality:
        feature_names += QUALITY_FEATURE_NAMES
    if want_controls:
        feature_names += STRUCTURE_FEATURE_NAMES

    # A peak list padded to exactly its bound may have been truncated
    # (src/data/utils.py:pad_peak truncates rather than dropping the molecule).
    # Width comes from the padded array itself, since the bound differs per run:
    # extractions over the cotrain training mixture apply
    # ckpt_meta.bounds_overrides(); ones embedding held-out datasets do not.
    h_width = (h_mask_all if h_mask_all is not None else h_shifts_all).shape[-1]
    c_width = (c_mask_all if c_mask_all is not None else c_shifts_all).shape[-1]
    n_h_truncated = n_c_truncated = 0

    rows = []
    for i in range(len(idx)):
        h_mask_i = h_mask_all[i].astype(bool) if h_mask_all is not None else np.isfinite(h_shifts_all[i])
        c_mask_i = c_mask_all[i].astype(bool) if c_mask_all is not None else np.isfinite(c_shifts_all[i])
        h_valid = h_shifts_all[i][h_mask_i]
        c_valid = c_shifts_all[i][c_mask_i]
        h_truncated = len(h_valid) >= h_width
        c_truncated = len(c_valid) >= c_width
        n_h_truncated += int(h_truncated)
        n_c_truncated += int(c_truncated)

        feats = {}
        feats.update(_one_axis_features("H", h_valid))
        feats.update(_one_axis_features("C", c_valid))
        n_h, n_c = feats["n_H_peaks"], feats["n_C_peaks"]
        feats["H_to_C_peak_ratio"] = float(n_h / n_c) if (n_h > 0 and n_c > 0) else np.nan
        feats["total_peaks"] = float(n_h + n_c)

        row = [feats[name] for name in SPECTRAL_FEATURE_NAMES]
        if h_nh_all is not None:
            nh_valid = h_nh_all[i][h_mask_i]
            nh_feats = _nh_features(nh_valid)
            row += [nh_feats["H_nH_mean"], nh_feats["H_nH_max"], nh_feats["H_nH_sum"], nh_feats["H_frac_CH"], nh_feats["H_frac_CH2"], nh_feats["H_frac_CH3"]]
        if h_mult_all is not None:
            mult_valid = h_mult_all[i][h_mask_i]
            mult_feats = _multiplet_features(mult_valid)
            row += [mult_feats["H_multiplet_n_unique"], mult_feats["H_multiplet_mode_frac"], mult_feats["H_multiplet_entropy"]]
        if want_quality:
            n_h_ref, n_c_unique, n_c_total = _structure_reference(smiles[i], h_reference)
            q = _quality_features(nh_feats["H_nH_sum"], feats["n_C_peaks"], n_h_ref, n_c_unique,
                                   h_truncated, c_truncated)
            row += [q[name] for name in QUALITY_FEATURE_NAMES]
            if want_controls:
                # Controls are structure-only: truncation of the peak list
                # doesn't invalidate them, so they are NOT gated on
                # h_truncated/c_truncated.
                row += [n_h_ref, n_c_total]
        rows.append(row)

    if n_h_truncated or n_c_truncated:
        print(f"[warn] Peak lists at the padding bound (possibly truncated): "
              f"{n_h_truncated}/{len(idx)} 1H (width {h_width}), "
              f"{n_c_truncated}/{len(idx)} 13C (width {c_width}). Their counts and "
              f"shift statistics understate the real spectrum"
              + ("; they are NaN'd out of the quality features." if want_quality else "."))

    return np.array(rows, dtype=np.float64), feature_names
