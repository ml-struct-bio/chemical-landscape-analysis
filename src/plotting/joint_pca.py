"""Figures for the joint structural + spectral PCA.

Consumes `data/04_joint_pca/<tag>/` and nothing else. Every molecule choice,
percentile range, correlation and normalization was already decided by
`src/analysis/joint_pca.py`; this module places ink.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.constants import DEFAULT_C_PPM_RANGE, DEFAULT_H_PPM_RANGE
from src.plotting.mol_render import mol_to_image
from src.plotting.spectra import C_COLOR, H_COLOR, draw_stick_spectrum
from src.plotting.style import raster, save_pdf


def _ordinal(pct: float) -> str:
    """'1st', '2nd', '99th'. The old scripts formatted these as f'{pct:.0f}th'
    and so rendered the default lower endpoint as '1th'."""
    n = int(round(pct))
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


class StepPeaks:
    """Ragged peak arrays for the traversal steps, in `traversal.csv` row order."""

    def __init__(self, path: Optional[Path]):
        self.ok = path is not None and path.exists()
        if not self.ok:
            return
        blob = np.load(path)
        self.h_values, self.h_offsets = blob["h_values"], blob["h_offsets"]
        self.c_values, self.c_offsets = blob["c_values"], blob["c_offsets"]
        self.nh_values = blob["h_nh_values"] if "h_nh_values" in blob.files else None

    def h(self, i: int) -> np.ndarray:
        return self.h_values[self.h_offsets[i]:self.h_offsets[i + 1]]

    def c(self, i: int) -> np.ndarray:
        return self.c_values[self.c_offsets[i]:self.c_offsets[i + 1]]

    def nh(self, i: int) -> Optional[np.ndarray]:
        if self.nh_values is None:
            return None
        return self.nh_values[self.h_offsets[i]:self.h_offsets[i + 1]]


def joint_traversal_figure(pc: int, steps: pd.DataFrame, stats: pd.Series, peaks: StepPeaks,
                           backdrop: np.ndarray, bars: Optional[pd.DataFrame], out_path: Path,
                           *, h_ppm_range: Tuple[float, float] = DEFAULT_H_PPM_RANGE,
                           c_ppm_range: Tuple[float, float] = DEFAULT_C_PPM_RANGE,
                           mol_size: int = 260, highlight_color: str = "red",
                           pct_lo: float = 1.0, pct_hi: float = 99.0) -> Path:
    """One figure per PC, rows top to bottom: 1H sticks, 13C sticks, structure,
    [feature bars], scatter.

    That order is deliberate -- spectrum in, structure out, matching the model's
    own direction. Every panel in a column is the SAME real molecule.
    """
    pc_idx, other_dim = pc - 1, int(stats["other_dim"]) - 1
    n_steps = len(steps)
    show_bars = bars is not None and len(bars) > 0
    bar_features = list(dict.fromkeys(bars["feature"])) if show_bars else []

    # The structure panel is a square image in a 2.1in-wide column, so its row
    # height is kept near that width; a taller row just pads it with whitespace.
    if show_bars:
        row_heights = [1.0, 1.0, 1.05, 1.15, 1.25]
        bar_row, scatter_row = 3, 4
    else:
        row_heights = [1.0, 1.0, 1.05, 1.25]
        bar_row, scatter_row = None, 3

    fig, axes = plt.subplots(len(row_heights), n_steps,
                             figsize=(n_steps * 2.1, 2.0 * sum(row_heights)),
                             squeeze=False, gridspec_kw={"height_ratios": row_heights})

    for i, (_, step) in enumerate(steps.iterrows()):
        row = int(step["peak_row"])
        draw_stick_spectrum(axes[0][i], peaks.h(row), h_ppm_range, H_COLOR,
                            heights=peaks.nh(row), label="$^{1}$H" if i == 0 else "",
                            y_max=float(stats["h_y_max"]))
        draw_stick_spectrum(axes[1][i], peaks.c(row), c_ppm_range, C_COLOR,
                            label="$^{13}$C" if i == 0 else "")
        axes[0][i].set_title(f"PC{pc}={step['pc_value']:.2f}\n{int(step['n_h_peaks'])} "
                             f"$^{{1}}$H / {int(step['n_c_peaks'])} $^{{13}}$C peaks", fontsize=6)
        axes[1][i].set_xlabel("ppm", fontsize=6)

        ax_mol = axes[2][i]
        img = mol_to_image(step["smiles"], size=mol_size)
        if img is not None:
            ax_mol.imshow(img)
        else:
            ax_mol.text(0.5, 0.5, "unparseable\nSMILES", ha="center", va="center",
                        fontsize=6, color="0.5", transform=ax_mol.transAxes)
        ax_mol.axis("off")
        if i == 0:
            # axis("off") hides the ylabel, so the row is labelled with text.
            ax_mol.text(-0.06, 0.5, "structure", fontsize=6, ha="right", va="center",
                        transform=ax_mol.transAxes)

        if show_bars:
            ax_bar = axes[bar_row][i]
            vals = bars[bars["step"] == step["step"]].set_index("feature")["value"]
            ax_bar.bar(range(len(bar_features)), [vals.get(f, np.nan) for f in bar_features])
            ax_bar.set_ylim(0, 1)
            ax_bar.set_xticks(range(len(bar_features)))
            ax_bar.set_xticklabels(bar_features, rotation=45, ha="right", fontsize=5)
            ax_bar.set_yticks([])
            if i == 0:
                ax_bar.set_ylabel("top-|r|\nfeatures", fontsize=6, rotation=0, ha="right",
                                  va="center", labelpad=10)

        ax = axes[scatter_row][i]
        raster(ax.scatter(backdrop[:, pc_idx], backdrop[:, other_dim], s=0.3, alpha=0.2,
                          c="lightgray"))
        ax.scatter(step["x"], step["y"], c=highlight_color, s=30, zorder=5)
        ax.set_xlabel(f"PC{pc}", fontsize=6)
        ax.set_ylabel(f"PC{other_dim + 1}", fontsize=6)
        ax.tick_params(labelsize=5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    bar_note = (f"\nBars = this molecule's {len(bar_features)} most PC{pc}-correlated spectral "
                f"features, each scaled to its own [{_ordinal(pct_lo)}, {_ordinal(pct_hi)}] "
                f"percentile across the corpus.") if show_bars else ""
    fig.suptitle(f"PC{pc} joint traversal (all other PCs fixed at median; range = "
                 f"[{_ordinal(pct_lo)}, {_ordinal(pct_hi)}] percentile). Spectra and structure "
                 f"are the SAME molecule at each step; $^{{1}}$H stick heights = nH.{bar_note}",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95 if show_bars else 0.96))
    return save_pdf(fig, out_path)


def correlation_grid(pc_values: np.ndarray, other_values: np.ndarray, best_df: pd.DataFrame,
                      name_col: str, out_path: Path, title_ylabel: bool = True,
                      alpha: float = 0.02) -> Path:
    """One scatter per PC against its single best-correlated column.

    `alpha` is low by design: this draws every molecule (millions on the train
    split), so anything high enough to see an individual point turns the panel
    into a solid blob and hides the density structure that is the point.
    """
    n_pcs = len(best_df)
    ncols = min(4, n_pcs)
    nrows = int(np.ceil(n_pcs / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.4 * ncols, 2.6 * nrows),
                             constrained_layout=True, squeeze=False)

    for i in range(nrows * ncols):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        if i >= n_pcs:
            ax.axis("off")
            continue
        row = best_df.iloc[i]
        name = row[name_col]
        if name is None or (isinstance(name, float) and np.isnan(name)):
            ax.axis("off")
            continue

        y = other_values[:, i]
        x = pc_values[:, i]
        mask = np.isfinite(x) & np.isfinite(y)
        raster(ax.scatter(x[mask], y[mask], alpha=alpha, s=2, linewidths=0))
        ax.set_title(f"{name}\nr={row['r']:.3f}", fontsize=9)
        ax.set_xlabel(f"PC{i + 1}")
        if title_ylabel:
            ax.set_ylabel(str(name))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

    return save_pdf(fig, out_path)


def correlation_heatmap(corr_df: pd.DataFrame, out_path: Path, title: str) -> Path:
    """The full PC x feature matrix, so the whole panel is visible at once
    rather than only each PC's single winner."""
    values = corr_df.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    vmax = max(float(np.abs(finite).max()) if finite.size else 1.0, 1e-6)

    fig, ax = plt.subplots(figsize=(0.34 * values.shape[1] + 3.0, 0.34 * values.shape[0] + 2.0))
    im = ax.imshow(values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(values.shape[1]))
    ax.set_xticklabels(corr_df.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(values.shape[0]))
    ax.set_yticklabels(corr_df.index, fontsize=7)
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8, label="r")
    fig.tight_layout()
    return save_pdf(fig, out_path)
