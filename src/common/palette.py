"""Shared, user-specifiable colors for every categorical scatter in the pipeline.

Why this exists
---------------
Color assignment used to be implicit and, worse, *positional*:

  * `pretty_plot_analysis.color_by_dataset` and `umap_plots.color_by_dataset`
    were two independent implementations, each assigning `tab10` by
    sorted-index -- so the same source dataset could be drawn in different
    colors by scripts 5/6 and script 7.
  * `color_by_cluster` built a colormap with ONE SLOT PER CLUSTER ID
    (`rainbow(cluster_id / (n_clusters - 1))`), so a cluster's color was a
    pure function of its numeric id. With 9,740 clusters that made
    numerically-close ids indistinguishable: on the cotrain-v3 corpus,
    clusters 0 and 31 both resolved to exactly `#7f00ff`, so highlighting the
    three largest clusters (0, 3988, 31) drew two of them in identical purple.
  * Scripts 8/14/15 each kept their own `QUALITATIVE_PALETTE[i % 10]`, keyed
    on enumeration order.

This module is the single source of truth. Colors come from a checked-in YAML
file (`configs/colors.yaml` by default) and may be overridden per-run from the
CLI. Nothing here depends on which script is calling.

Key design points
-----------------
* **Cluster colors are keyed by cluster id, not by id position.** The CLI form
  is two parallel lists (`--highlight-clusters 0 3988 31 --cluster-colors A B
  C`), which is zipped into an `{id: color}` mapping -- the same shape the YAML
  file uses -- so the two inputs are interchangeable and color no longer
  depends on how far apart the ids happen to be.
* **Unknown keys are an error, not a silent no-op.** Naming a label that isn't
  in the data (`nmrexp-v3` when the data says `nmrexp`) raises with the valid
  labels listed. A palette that silently does nothing produces a figure that
  looks fine and is wrong.
* **Partial palettes are fine.** Labels you don't name keep an automatic color,
  chosen to skip every hue you explicitly claimed, so your choices can never
  collide with a fallback.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # matplotlib is present everywhere the plotting layer runs
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - headless/import-light contexts
    mcolors = None
    plt = None


# Default config location, relative to the repo root.
DEFAULT_PALETTE_PATH = Path("configs/colors.yaml")

# The automatic fallback ramp. tab10 in a fixed, explicit order so the
# "skip hues the user already claimed" logic below is reproducible and does not
# silently change if matplotlib reorders its colormaps.
DEFAULT_QUALITATIVE: Tuple[str, ...] = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
)

# Everything not highlighted / not in the top-N: the pipeline-wide background.
DEFAULT_OTHER_COLOR = "#bfbfbf"

# The color modes a palette can carry. `cluster` is keyed by int, the rest by str.
COLOR_MODES: Tuple[str, ...] = ("dataset", "cluster", "np_class", "sim_real", "anchor")


def to_rgba(color: Any) -> Tuple[float, float, float, float]:
    """Any matplotlib-acceptable color -> an RGBA 4-tuple.

    Accepts hex ('#1f77b4'), named colors ('teal'), grayscale strings ('0.75'),
    and existing RGB/RGBA sequences, so a palette file can use whichever form
    reads best without callers caring.
    """
    if mcolors is None:  # pragma: no cover
        raise RuntimeError("matplotlib is required to resolve colors")
    try:
        return tuple(float(v) for v in mcolors.to_rgba(color))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{color!r} is not a valid matplotlib color") from exc


def _hue_of(color: Any) -> float:
    r, g, b, _ = to_rgba(color)
    return float(mcolors.rgb_to_hsv((r, g, b))[0])


def _fallback_colors(n: int, claimed: Sequence[Any]) -> List[Tuple[float, float, float, float]]:
    """`n` automatic colors that avoid every hue in `claimed`.

    Without this, explicitly asking for '#1f77b4' on one dataset and letting
    another fall back would hand the second the same tab10 blue -- the exact
    collision this module exists to prevent.
    """
    claimed_hues = [_hue_of(c) for c in claimed]

    def far_enough(candidate: str) -> bool:
        h = _hue_of(candidate)
        # Hue is circular, so compare on the shorter arc.
        return all(min(abs(h - ch), 1.0 - abs(h - ch)) > 0.04 for ch in claimed_hues)

    pool = [c for c in DEFAULT_QUALITATIVE if far_enough(c)]
    if len(pool) < n:
        # More categories than distinct spare hues: fall back to an evenly
        # spaced sweep, which at least stays maximally separated among itself.
        if plt is None:  # pragma: no cover
            pool = list(DEFAULT_QUALITATIVE)
        else:
            pool = [mcolors.to_hex(c) for c in plt.cm.hsv(np.linspace(0, 0.95, max(n, 1)))]
    return [to_rgba(pool[i % len(pool)]) for i in range(n)]


@dataclass
class Palette:
    """Resolved color overrides, one mapping per color mode.

    Values are whatever the user wrote (hex/name/tuple); they are converted to
    RGBA lazily by `resolve`, so an invalid color is reported against the mode
    and key that carry it.
    """

    dataset: Dict[str, Any] = field(default_factory=dict)
    cluster: Dict[int, Any] = field(default_factory=dict)
    np_class: Dict[str, Any] = field(default_factory=dict)
    sim_real: Dict[str, Any] = field(default_factory=dict)
    anchor: Dict[str, Any] = field(default_factory=dict)
    other: Any = DEFAULT_OTHER_COLOR

    def mode(self, name: str) -> Dict[Any, Any]:
        if name not in COLOR_MODES:
            raise KeyError(f"Unknown color mode {name!r}; expected one of {list(COLOR_MODES)}")
        return getattr(self, name)

    def merged_with(self, other: "Palette") -> "Palette":
        """`other` wins key-by-key -- used to layer CLI flags over the file."""
        out = Palette(other=other.other if other.other != DEFAULT_OTHER_COLOR else self.other)
        for name in COLOR_MODES:
            merged = dict(self.mode(name))
            merged.update(other.mode(name))
            setattr(out, name, merged)
        return out

    def is_empty(self) -> bool:
        return all(not self.mode(n) for n in COLOR_MODES)


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------


def load_palette_file(path: Optional[Path], root_dir: Optional[Path] = None) -> Palette:
    """Reads a palette YAML/JSON file. A missing DEFAULT path is not an error
    (the pipeline just uses automatic colors); a missing EXPLICIT path is."""
    explicit = path is not None
    if path is None:
        path = (root_dir or Path.cwd()) / DEFAULT_PALETTE_PATH
    path = Path(path)

    if not path.exists():
        if explicit:
            raise FileNotFoundError(f"--palette {path} does not exist.")
        return Palette()

    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        import yaml

        raw = yaml.safe_load(text) or {}
    else:
        import json

        raw = json.loads(text) or {}
    return palette_from_dict(raw, source=str(path))


def palette_from_dict(raw: Dict[str, Any], source: str = "<dict>") -> Palette:
    unknown = [k for k in raw if k not in COLOR_MODES and k != "other"]
    if unknown:
        raise ValueError(
            f"{source}: unknown top-level key(s) {unknown}. "
            f"Expected any of {list(COLOR_MODES) + ['other']}."
        )
    pal = Palette(other=raw.get("other", DEFAULT_OTHER_COLOR))
    for name in COLOR_MODES:
        block = raw.get(name) or {}
        if not isinstance(block, dict):
            raise ValueError(
                f"{source}: '{name}' must be a mapping of label -> color, got {type(block).__name__}."
            )
        if name == "cluster":
            setattr(pal, name, {int(k): v for k, v in block.items()})
        else:
            setattr(pal, name, {str(k): v for k, v in block.items()})
    return pal


# -----------------------------------------------------------------------------
# CLI plumbing
# -----------------------------------------------------------------------------

_PAIR_RE = re.compile(r"^(?P<key>[^=]+)=(?P<color>.+)$")


def parse_pairs(values: Optional[Sequence[str]], mode: str) -> Dict[str, Any]:
    """`['nmrexp=#1f77b4', 'uspto=grey']` -> `{'nmrexp': '#1f77b4', ...}`."""
    out: Dict[str, Any] = {}
    for item in values or []:
        m = _PAIR_RE.match(item.strip())
        if not m:
            raise ValueError(
                f"--{mode.replace('_', '-')}-colors entry {item!r} is not of the form NAME=COLOR "
                f"(e.g. nmrexp='#1f77b4')."
            )
        key, color = m.group("key").strip(), m.group("color").strip()
        to_rgba(color)  # validate eagerly, so a typo fails now rather than at draw time
        out[key] = color
    return out


def zip_sequence(keys: Optional[Sequence[Any]], colors: Optional[Sequence[str]],
                  keys_flag: str, colors_flag: str) -> Dict[Any, Any]:
    """Pairs two parallel CLI lists into a mapping, e.g.
    `--highlight-clusters 0 3988 31` with `--cluster-colors A B C`.

    Length mismatch is the failure mode this form invites (edit one list,
    forget the other), so it is checked explicitly and reported with both
    lengths rather than silently zipping to the shorter one.
    """
    if not colors:
        return {}
    if not keys:
        raise ValueError(f"{colors_flag} was given without {keys_flag}; there is nothing to pair the colors with.")
    if len(keys) != len(colors):
        raise ValueError(
            f"{keys_flag} has {len(keys)} entr{'y' if len(keys) == 1 else 'ies'} "
            f"but {colors_flag} has {len(colors)}. They are paired positionally, so "
            f"they must be the same length.\n"
            f"  {keys_flag}: {list(keys)}\n"
            f"  {colors_flag}: {list(colors)}"
        )
    for c in colors:
        to_rgba(c)
    return dict(zip(keys, colors))


def add_palette_args(parser, modes: Sequence[str] = COLOR_MODES) -> None:
    """Adds `--palette` plus one `--<mode>-colors` flag per requested mode.

    Every flag is cosmetic, so all of them are safe to change with
    `--plot-only` on the scripts that support it.
    """
    group = parser.add_argument_group("colors")
    group.add_argument(
        "--palette", type=Path, default=None,
        help=f"Palette file (YAML or JSON) mapping labels to colors. "
             f"Default: {DEFAULT_PALETTE_PATH} if it exists, else automatic colors. "
             f"CLI --*-colors flags override individual entries.")
    group.add_argument(
        "--other-color", type=str, default=None,
        help="Color for everything not highlighted / bucketed into 'Other'. "
             f"Default: {DEFAULT_OTHER_COLOR}.")

    if "cluster" in modes:
        group.add_argument(
            "--cluster-colors", nargs="+", default=None, metavar="COLOR",
            help="Colors paired POSITIONALLY with --highlight-clusters, e.g. "
                 "--highlight-clusters 0 3988 31 --cluster-colors '#e6194b' '#4363d8' '#3cb44b'. "
                 "Must have the same number of entries as --highlight-clusters. "
                 "Omit to use automatic, well-separated colors.")
    for mode in ("dataset", "np_class", "sim_real", "anchor"):
        if mode not in modes:
            continue
        flag = f"--{mode.replace('_', '-')}-colors"
        group.add_argument(
            flag, nargs="+", default=None, metavar="NAME=COLOR",
            help=f"Per-{mode} colors as NAME=COLOR pairs, e.g. "
                 f"{flag} nmrexp='#1f77b4' uspto=grey. Overrides the palette file. "
                 f"Naming a label that is not present in the data is an error.")


def palette_from_args(args, root_dir: Optional[Path] = None,
                       cluster_keys: Optional[Sequence[int]] = None) -> Palette:
    """File first, then CLI overrides on top.

    `cluster_keys` is what `--cluster-colors` gets zipped against -- normally
    `args.highlight_clusters`.
    """
    base = load_palette_file(getattr(args, "palette", None), root_dir=root_dir)

    override = Palette()
    for mode in ("dataset", "np_class", "sim_real", "anchor"):
        values = getattr(args, f"{mode}_colors", None)
        if values:
            setattr(override, mode, parse_pairs(values, mode))

    cluster_colors = getattr(args, "cluster_colors", None)
    if cluster_colors:
        keys = cluster_keys if cluster_keys is not None else getattr(args, "highlight_clusters", None)
        override.cluster = zip_sequence(
            [int(k) for k in keys] if keys else None, cluster_colors,
            "--highlight-clusters", "--cluster-colors")

    other = getattr(args, "other_color", None)
    if other:
        to_rgba(other)
        override.other = other

    return base.merged_with(override)


# -----------------------------------------------------------------------------
# Resolution
# -----------------------------------------------------------------------------


def resolve(labels: Sequence[Any], overrides: Optional[Dict[Any, Any]], *,
             mode: str, ordered_keys: Optional[Sequence[Any]] = None,
             ) -> Dict[Any, Tuple[float, float, float, float]]:
    """label -> RGBA for every distinct label present in the data.

    `overrides` wins; anything unnamed gets an automatic color chosen to avoid
    the hues that were named. An override key that does not occur in `labels`
    raises, listing what is actually there -- see the module docstring.
    """
    present = list(ordered_keys) if ordered_keys is not None else sorted({l for l in labels})
    overrides = dict(overrides or {})

    unknown = [k for k in overrides if k not in present]
    if unknown:
        raise ValueError(
            f"--{mode.replace('_', '-')}-colors / palette '{mode}' names "
            f"{unknown}, which do not appear in the data.\n"
            f"Available {mode} labels: {present}"
        )

    for key, color in overrides.items():
        try:
            to_rgba(color)
        except ValueError as exc:
            raise ValueError(f"palette '{mode}' entry {key!r}: {exc}") from None

    unnamed = [p for p in present if p not in overrides]
    auto = _fallback_colors(len(unnamed), list(overrides.values()))
    auto_iter = iter(auto)
    return {p: (to_rgba(overrides[p]) if p in overrides else next(auto_iter)) for p in present}


def colors_for(labels: Sequence[Any], palette_map: Dict[Any, Tuple[float, float, float, float]]
                ) -> np.ndarray:
    """Per-point RGBA array, in the order `labels` came in."""
    return np.array([palette_map[l] for l in labels])


def entity_colors(names: Sequence[str], palette: Optional[Palette] = None
                   ) -> Dict[str, Tuple[float, float, float, float]]:
    """name -> RGBA for the named entities drawn by scripts 8, 14 and 15
    (real/synthetic pairs, anchor natural products, external dataset specs).

    Those three used to index `QUALITATIVE_PALETTE[i % 10]` by enumeration
    order, so a color meant nothing across runs and silently reshuffled when
    an entity was added, removed, or failed to resolve. Keying on the entity's
    own name makes it stable and hand-selectable.
    """
    names = [str(n) for n in names]
    return resolve(names, (palette or Palette()).anchor, mode="anchor", ordered_keys=names)
