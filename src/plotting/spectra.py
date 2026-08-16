"""Stick-spectrum drawing."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


H_COLOR = "#1f77b4"
C_COLOR = "#d62728"


def resolve_stick_heights(peaks: np.ndarray,
                          heights: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Drops non-finite peaks; defaults heights to 1.0 where missing or
    non-positive."""
    peaks = np.asarray(peaks, dtype=float)
    finite = np.isfinite(peaks)
    peaks = peaks[finite]
    if heights is None:
        return peaks, np.ones_like(peaks)
    heights = np.asarray(heights, dtype=float)[finite]
    return peaks, np.where(np.isfinite(heights) & (heights > 0), heights, 1.0)


def draw_stick_spectrum(ax, peaks: np.ndarray, ppm_range: Tuple[float, float], color: str,
                        heights: Optional[np.ndarray] = None, label: str = "",
                        fontsize: int = 6, y_max: Optional[float] = None) -> None:
    """A minimal but faithful stick spectrum: one vertical line per peak at its
    real ppm position, on a reversed (NMR-convention) axis clamped to a FIXED
    `ppm_range` so every panel of a filmstrip is directly comparable.

    `heights` (the 1H integration, `h_peak_nH`) sets stick heights when
    available, otherwise every peak is unit height. `y_max` pins the vertical
    scale -- callers pass the filmstrip-wide maximum so a tall peak in one panel
    does not make the others look different than they are.

    No lineshape, broadening or J-coupling is simulated: what is drawn is
    exactly the peak list the model was conditioned on, nothing interpolated.
    """
    lo, hi = ppm_range
    peaks, heights = resolve_stick_heights(peaks, heights)

    if y_max is None:
        y_max = float(heights.max()) if len(heights) else 1.0
    y_max = max(float(y_max), 1e-9)
    if len(peaks):
        ax.vlines(peaks, 0.0, heights, color=color, linewidth=0.9)
    ax.axhline(0.0, color="0.75", linewidth=0.5)

    ax.set_xlim(hi, lo)  # reversed: high ppm on the left
    ax.set_ylim(0.0, y_max * 1.15)
    ax.set_yticks([])
    ax.tick_params(labelsize=fontsize - 1)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    if label:
        ax.set_ylabel(label, fontsize=fontsize, rotation=0, ha="right", va="center", labelpad=10)

    # Peaks outside the plotted window would silently vanish; say so instead.
    n_clipped = int(((peaks < lo) | (peaks > hi)).sum())
    if n_clipped:
        ax.text(0.02, 0.85, f"{n_clipped} off-scale", transform=ax.transAxes,
                fontsize=fontsize - 1, color="0.4")
