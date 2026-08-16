#!/usr/bin/env python
"""
05_umap.py
==========

Figures from a stored UMAP projection. Reads `data/05_umap/<tag>/` and writes
PDFs to `figures/05_umap/<tag>/`.

Every colouring is a plot-time choice over columns the analysis already stored,
so one invocation can emit one figure or a hundred without ever refitting.

Colourings (`--color-by`)
-------------------------
    none            all one colour (grey by default; set with --point-color)
    dataset         source dataset of the cotrain mixture
    sim_real        real vs. synthetic vs. unknown
    split           train / val / test
    cluster         Butina cluster id (use --highlight to pick a few)
    <property>      any RDKit property the analysis computed, e.g. MolWt
    <spectral>      any NMR spectral feature, e.g. n_C_peaks
    all-properties  one figure per molecular property
    all-spectral    one figure per spectral feature
    all             everything above

Colours
-------
Categorical colourings resolve through `configs/colors.yaml` plus the
`--dataset-colors` / `--cluster-colors` / `--sim-real-colors` flags, so a label
keeps the same colour in every figure of the pipeline. Palette entries naming a
label absent from a given figure are ignored with a note rather than failing --
Butina renumbers clusters on every re-run, so a pinned cluster id goes stale.

Usage
-----
    python plotting/05_umap.py --tag global_cond --color-by dataset
    python plotting/05_umap.py --tag global_cond --color-by cluster \
        --highlight 0 3988 31 --cluster-colors '#e6194b' '#4363d8' '#3cb44b'
    python plotting/05_umap.py --tag ecfp --color-by all-properties
    python plotting/05_umap.py --all-tags --color-by none dataset sim_real
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402

import pandas as pd  # noqa: E402

from src.common.manifest import require_manifest  # noqa: E402
from src.common.palette import add_palette_args, palette_from_args, resolve  # noqa: E402
from src.common.paths import DATA_ROOT, figures_dir  # noqa: E402
from src.plotting.style import apply_style  # noqa: E402
from src.plotting.umap_figures import (  # noqa: E402
    DEFAULT_PLAIN_COLOR,
    POINT_ALPHA,
    POINT_SIZE,
    _relevant_overrides,
    plot_anchors,
    plot_categorical,
    plot_continuous,
    plot_neighbor_similarity,
    plot_plain,
    plot_with_insets,
)


def _read_csv(path: Path):
    """A table the analysis may or may not have written."""
    return pd.read_csv(path) if path.exists() else None

SLUG = "05_umap"
SCHEMA_VERSION = 1

# color-by name -> the palette mode its colours resolve through, or None for
# "no configurable palette". `split` has none: it is train/val/test, not a
# scientific category anyone pins a colour for, and giving it the `dataset`
# namespace would make every dataset entry in colors.yaml look misapplied.
CATEGORICAL = {
    "dataset": "dataset",
    "sim_real": "sim_real",
    "cluster": "cluster",
    "split": None,
}

# (max points, marker size, alpha). A UMAP of this corpus can be 300 molecules or
# 2.5 million, and no single (size, alpha) reads well across four orders of
# magnitude: values that show density in the large case are invisible in the
# small one.
#
# This is NOT the default. The previous pipeline drew every UMAP at a flat
# 0.25/0.25 and the figures are matched to it, so the ramp is opt-in via
# --auto-point-style -- worth reaching for on a `test`-split tag, where 0.25pt
# markers leave a near-empty page.
AUTO_POINT_STYLE = (
    (1_000, 14.0, 0.90),
    (10_000, 6.0, 0.70),
    (100_000, 2.0, 0.50),
    (1_000_000, 1.0, 0.35),
    (float("inf"), 0.5, 0.30),
)


def auto_point_style(n: int) -> tuple:
    for limit, size, alpha in AUTO_POINT_STYLE:
        if n <= limit:
            return size, alpha
    return AUTO_POINT_STYLE[-1][1:]


def parse_args():
    p = argparse.ArgumentParser(
        description="Draw UMAP figures from a stored projection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--tag", type=str, default="global_cond",
                   help="Which projection to draw, i.e. data/05_umap/<tag>/.")
    p.add_argument("--all-tags", action="store_true",
                   help="Draw every tag under data/05_umap/.")
    p.add_argument("--color-by", nargs="+", default=["none"],
                   help="One or more colourings; see the module docstring.")
    p.add_argument("--list-colorings", action="store_true",
                   help="Print what this tag can be coloured by, then exit.")

    p.add_argument("--highlight", nargs="+", default=None,
                   help="For a categorical colouring, colour only these labels and grey "
                        "the rest (e.g. --color-by cluster --highlight 0 3988 31).")
    p.add_argument("--point-color", type=str, default=DEFAULT_PLAIN_COLOR,
                   help="Colour for --color-by none, and the backdrop of the anchor "
                        "and inset figures.")
    p.add_argument("--point-size", type=float, default=None,
                   help=f"Default: {POINT_SIZE}, the previous pipeline's value.")
    p.add_argument("--alpha", type=float, default=None,
                   help=f"Default: {POINT_ALPHA}, the previous pipeline's value.")
    p.add_argument("--auto-point-style", action="store_true",
                   help="Scale marker size and alpha to the number of points drawn (see "
                        "AUTO_POINT_STYLE) instead of the flat defaults. Useful on a "
                        "small tag, where 0.25pt markers are effectively invisible.")
    p.add_argument("--highlight-size", type=float, default=None,
                   help="Marker size for highlighted points. Default: 8x --point-size.")
    p.add_argument("--cmap", type=str, default="viridis",
                   help="Colormap for continuous colourings.")
    p.add_argument("--clip-percentiles", type=float, nargs=2, default=[1.0, 99.0],
                   help="Robust colour range for continuous colourings. Long-tailed "
                        "properties would otherwise map every ordinary molecule to one "
                        "end of the colormap.")
    p.add_argument("--max-points", type=int, default=0,
                   help="Draw at most this many points, sampled with --seed. 0 draws all. "
                        "Purely a display choice -- the stored projection is untouched.")
    p.add_argument("--seed", type=int, default=1234)

    a = p.add_argument_group("anchors and insets")
    a.add_argument("--insets", action="store_true",
                   help="Draw an inset per region (anchors and/or --inset-clusters from "
                        "the analysis): a zoomed scatter beside a grid of that region's "
                        "molecules, with the region boxed on the main map.")
    a.add_argument("--no-neighbors", action="store_true",
                   help="Mark anchors only, without their cosine nearest neighbours.")
    a.add_argument("--anchor-size", type=float, default=90.0,
                   help="Marker size for anchor molecules.")
    a.add_argument("--neighbor-size", type=float, default=22.0,
                   help="Marker size for an anchor's nearest neighbours.")
    a.add_argument("--max-insets", type=int, default=6,
                   help="Inset rows drawn; extra regions are reported and skipped.")
    a.add_argument("--mols-per-row", type=int, default=3,
                   help="Molecules per row inside an inset's structure grid.")
    add_palette_args(p, modes=("dataset", "cluster", "sim_real", "anchor"))
    return p.parse_args()


def _load(directory: Path):
    coords = np.load(directory / "coords.npz")["coords"]
    cat = np.load(directory / "categorical.npz", allow_pickle=False)
    props = np.load(directory / "properties.npz", allow_pickle=False)
    spec = np.load(directory / "spectral.npz", allow_pickle=False)
    return coords, cat, props, spec


def _categorical_values(cat, name: str):
    """Per-point labels for a categorical column, decoded from codes+levels."""
    if name == "cluster":
        return cat["cluster"] if "cluster" in cat.files else None
    codes_key, levels_key = f"{name}_codes", f"{name}_levels"
    if codes_key not in cat.files:
        return None
    levels = [str(x) for x in cat[levels_key]]
    return np.asarray(levels, dtype=object)[cat[codes_key]]


def expand_colorings(requested, props_names, spec_names, cat) -> list:
    out = []
    for item in requested:
        if item == "all":
            out += ["none"] + [k for k in CATEGORICAL if _categorical_values(cat, k) is not None]
            out += list(props_names) + list(spec_names)
        elif item == "all-properties":
            out += list(props_names)
        elif item == "all-spectral":
            out += list(spec_names)
        else:
            out.append(item)
    seen = set()
    return [x for x in out if not (x in seen or seen.add(x))]


def draw_tag(tag: str, args) -> None:
    directory, manifest = require_manifest(SLUG, tag, schema_version=SCHEMA_VERSION)
    params = manifest.get("params", {})
    out_dir = figures_dir(SLUG, tag, create=True)
    coords, cat, props, spec = _load(directory)

    props_names = [str(x) for x in props["names"]]
    spec_names = [str(x) for x in spec["names"]]
    n_total = len(coords)

    print(f"\n### {tag} -- {params.get('embedding', '?')} ({n_total:,} molecules) ###")

    if args.list_colorings:
        available = [k for k in CATEGORICAL if _categorical_values(cat, k) is not None]
        print(f"  categorical : none, {', '.join(available)}")
        print(f"  properties  : {', '.join(props_names) or '(none)'}")
        print(f"  spectral    : {', '.join(spec_names) or '(none)'}")
        return

    # Display thinning. The stored projection keeps every point; this only
    # affects what gets drawn, and is seeded so figures stay comparable.
    idx = np.arange(n_total)
    if 0 < args.max_points < n_total:
        idx = np.sort(np.random.default_rng(args.seed).choice(n_total, args.max_points,
                                                              replace=False))
        print(f"  drawing {len(idx):,}/{n_total:,} points (--max-points)")
    xy = coords[idx]

    base_size, base_alpha = (auto_point_style(len(idx)) if args.auto_point_style
                             else (POINT_SIZE, POINT_ALPHA))
    point_size = args.point_size if args.point_size is not None else base_size
    alpha = args.alpha if args.alpha is not None else base_alpha

    palette = palette_from_args(args, root_dir=_REPO_ROOT)
    colorings = expand_colorings(args.color_by, props_names, spec_names, cat)
    label = params.get("embedding", tag)
    highlight_used: list = []

    for name in colorings:
        if name == "none":
            path = plot_plain(xy, out_dir / "umap_plain.pdf",
                              color=args.point_color, point_size=point_size,
                              alpha=alpha)
        elif name in CATEGORICAL:
            values = _categorical_values(cat, name)
            if values is None:
                print(f"[warn] '{name}' is not available for this tag -- skipping. "
                      f"(Was analysis/05_umap.py run with --cluster-tag?)")
                continue
            # --highlight names labels of ONE column, but a run can ask for
            # several colourings at once. Applying it where the labels do not
            # occur would abort the whole run over a column the user never meant
            # it for, so it is applied where it fits and skipped where it does
            # not; never matching at all is still an error (see below).
            highlight = args.highlight
            if highlight:
                if name == "cluster":
                    try:
                        highlight = [int(h) for h in highlight]
                    except ValueError:
                        highlight = None
                if highlight is not None and set(highlight) <= set(np.unique(values).tolist()):
                    highlight_used.append(name)
                else:
                    highlight = None
            mode = CATEGORICAL[name]
            path = plot_categorical(
                xy, values[idx], out_dir / f"umap_{name}.pdf",
                mode=mode or "dataset",
                palette=palette if mode else None, highlight=highlight,
                point_size=point_size, alpha=alpha,
                highlight_size=args.highlight_size)
        elif name in props_names or name in spec_names:
            source, names = (props, props_names) if name in props_names else (spec, spec_names)
            values = source["values"][:, names.index(name)]
            path = plot_continuous(
                xy, values[idx], out_dir / f"umap_{name}.pdf",
                label=name, cmap=args.cmap,
                point_size=point_size, alpha=alpha,
                pct_lo=args.clip_percentiles[0], pct_hi=args.clip_percentiles[1])
        else:
            raise SystemExit(
                f"Unknown --color-by {name!r}.\n"
                f"Categorical: none, {', '.join(CATEGORICAL)}\n"
                f"Properties : {', '.join(props_names) or '(none)'}\n"
                f"Spectral   : {', '.join(spec_names) or '(none)'}\n"
                f"Run with --list-colorings to see what this tag supports.")
        print(f"Saved {path}")

    # --- anchors / insets -----------------------------------------------------
    anchors = _read_csv(directory / "anchors.csv")
    neighbors = _read_csv(directory / "anchor_neighbors.csv")
    regions = _read_csv(directory / "regions.csv")
    region_mols = _read_csv(directory / "region_mols.csv")

    if anchors is not None:
        matched = anchors[anchors["row"] >= 0]
        missing = anchors[anchors["row"] < 0]["anchor"].tolist()
        if missing:
            print(f"  [note] anchors not found in this corpus, not drawn: {missing}")
        if len(matched):
            names = matched["anchor"].tolist()
            color_map = resolve(names, _relevant_overrides(palette.mode("anchor"), names),
                                mode="anchor", ordered_keys=names)
            path = plot_anchors(
                xy, out_dir / "umap_anchors.pdf", anchors=matched,
                neighbors=None if args.no_neighbors else neighbors,
                color_map=color_map, base_color=args.point_color,
                point_size=point_size, alpha=alpha,
                show_neighbors=not args.no_neighbors)
            print(f"Saved {path}")

            if neighbors is not None and len(neighbors) and not args.no_neighbors:
                path = plot_neighbor_similarity(
                    neighbors, out_dir / "anchor_neighbor_similarity.pdf",
                    title=f"Cosine similarity to anchor — {label}", color_map=color_map)
                print(f"Saved {path}")

    if args.insets:
        if regions is None or not len(regions):
            print("[warn] --insets asked for, but this analysis stored no regions. "
                  "Re-run analysis/05_umap.py with --highlight-smiles and/or "
                  "--inset-clusters / --inset-top-clusters.")
        else:
            labels = regions["label"].tolist()
            region_colors = resolve(labels, _relevant_overrides(palette.mode("anchor"), labels),
                                    mode="anchor", ordered_keys=labels)
            path = plot_with_insets(
                xy, out_dir / "umap_insets.pdf",
                regions=regions, region_mols=region_mols, color_map=region_colors,
                anchors=anchors[anchors["row"] >= 0] if anchors is not None else None,
                neighbors=None if args.no_neighbors else neighbors,
                base_color=args.point_color, point_size=point_size, alpha=alpha,
                mols_per_row=args.mols_per_row, max_insets=args.max_insets)
            print(f"Saved {path}")

    if args.highlight and not highlight_used:
        categorical_drawn = [c for c in colorings if c in CATEGORICAL]
        raise SystemExit(
            f"--highlight {args.highlight} matched no labels in any categorical colouring "
            f"drawn ({categorical_drawn or 'none'}).\n"
            f"Highlighting names labels of one column -- e.g. "
            f"--color-by cluster --highlight 0 3988 31, or "
            f"--color-by dataset --highlight nmrexp.")
    elif args.highlight:
        print(f"  --highlight applied to: {', '.join(highlight_used)}")


def main():
    args = parse_args()
    apply_style()

    if args.all_tags:
        root = DATA_ROOT / SLUG
        tags = sorted(p.name for p in root.iterdir() if p.is_dir()) if root.exists() else []
        if not tags:
            raise SystemExit(f"No analysis output under {root}.\n"
                             f"Run: python analysis/{SLUG}.py --data-dir ... --embeddings all")
    else:
        tags = [args.tag]

    for tag in tags:
        draw_tag(tag, args)
    print(f"\nDone. {len(tags)} tag(s) -> figures/{SLUG}/")


if __name__ == "__main__":
    main()
