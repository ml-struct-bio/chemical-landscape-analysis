"""Nearest-molecule search for traversals.

This lived in the old `pca_traversal_analysis.py` and was imported by four
plotting modules, which meant the search ran at draw time. It is a search, so it
belongs here and its *result* -- the chosen molecule indices -- is what gets
written to `data/`.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def nearest_molecule_index(query_2d: np.ndarray, points_2d: np.ndarray) -> int:
    """Index of the nearest real molecule to `query_2d` by Euclidean distance.

    Computed ONLY in the 2 dimensions actually shown in a traversal panel's
    scatter, never the full PC space. That is deliberate and load-bearing: the
    figure asserts "this is the molecule sitting at that red dot", and a
    nearest neighbour in 8-D can be visibly far from the dot in the 2-D the
    reader is looking at.
    """
    dists = np.linalg.norm(points_2d - query_2d, axis=1)
    return int(np.argmin(dists))


def traversal_steps(pcs: np.ndarray, pc_idx: int, n_steps: int,
                    pct_lo: float = 1.0, pct_hi: float = 99.0) -> Tuple[np.ndarray, np.ndarray, int]:
    """The pipeline-wide traversal convention, in one place.

    Returns `(pc_values, step_indices, other_dim)`:

    * `pc_values` are `n_steps` points spanning the [pct_lo, pct_hi]
      PERCENTILE range of PC `pc_idx` -- **not** true min/max, which a single
      outlier would stretch into a range where every interior step lands in the
      same dense blob.
    * every other PC is held at its median, and `other_dim` is the second axis
      drawn in the scatter row.
    * `step_indices` point at REAL molecules: each step is snapped to the
      nearest actual molecule's own coordinates, never an interpolated point
      that no molecule occupies.
    """
    n_pcs = pcs.shape[1]
    medians = np.median(pcs, axis=0)
    lo, hi = np.percentile(pcs[:, pc_idx], [pct_lo, pct_hi])
    pc_values = np.linspace(lo, hi, n_steps)

    other_dim = (pc_idx + 1) % n_pcs if n_pcs > 1 else pc_idx
    points_2d = pcs[:, [pc_idx, other_dim]]

    step_indices = np.array([
        nearest_molecule_index(np.array([val, medians[other_dim]]), points_2d)
        for val in pc_values
    ], dtype=np.int64)
    return pc_values, step_indices, other_dim
