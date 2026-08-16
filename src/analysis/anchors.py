"""Anchor molecules: matching them, finding their neighbours, and the regions
and representative molecules an inset needs.

Everything a highlight/inset figure shows is decided here:

* which corpus row a queried SMILES actually is (canonical match, not string
  equality -- the same molecule is written many ways);
* its k nearest neighbours **by cosine similarity in the high-dimensional
  embedding**, not in the 2-D map. Those are the real neighbours in the space
  being studied; where they land on the map is the interesting part, and
  computing them from the 2-D coordinates would beg the question by returning a
  tight visual blob every time;
* the 2-D region an inset zooms into, and which molecules it draws.

The plotting side then only places ink. Computation only -- no matplotlib.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from rdkit import Chem
from rdkit import RDLogger


def canonicalize(smi: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol is not None else None


# A name is plain text: letters, digits, spaces, underscore, hyphen, dot. That
# deliberately excludes every character SMILES uses structurally, so
# `name=SMILES` can be told apart from a bare SMILES whose '=' is a double bond
# (`CC(=O)O` must not parse as name "CC(" ).
_ANCHOR_NAME_RE = re.compile(r"^[A-Za-z0-9 _.\-]+$")


def parse_anchor_specs(entries: Sequence[str]) -> List[Tuple[str, str]]:
    """`"name=SMILES"` or bare `"SMILES"` -> [(name, smiles)].

    Naming anchors matters once there is more than one: the legend, the inset
    titles and the neighbour table all key off it, and a raw SMILES makes an
    unreadable label.
    """
    out = []
    for i, entry in enumerate(entries):
        entry = entry.strip()
        if not entry:
            continue
        if "=" in entry:
            head, tail = entry.split("=", 1)
            if _ANCHOR_NAME_RE.match(head.strip()) and tail.strip():
                out.append((head.strip(), tail.strip()))
                continue
        out.append((f"anchor{i + 1}", entry))
    return out


def read_anchor_file(path: Path) -> List[Tuple[str, str]]:
    """One `name=SMILES` or bare SMILES per line; `#` comments ignored."""
    lines = [ln.strip() for ln in Path(path).read_text().splitlines()]
    return parse_anchor_specs([ln for ln in lines if ln and not ln.startswith("#")])


def match_anchors(specs: Sequence[Tuple[str, str]], corpus_smiles: Sequence[str],
                  n_workers: int = 1) -> pd.DataFrame:
    """Locate each anchor in the corpus by canonical SMILES.

    Unmatched anchors are reported and kept in the table with `row = -1` rather
    than dropped, so a figure that silently lost one is impossible to mistake
    for a figure that never had it.
    """
    from src.analysis.corpus import _canonicalize_many

    RDLogger.DisableLog("rdApp.*")
    wanted = {}
    for name, smiles in specs:
        canon = canonicalize(smiles)
        if canon is None:
            print(f"[warn] anchor {name!r}: RDKit cannot parse {smiles!r} -- skipping.")
            continue
        wanted.setdefault(canon, []).append((name, smiles))

    if not wanted:
        return pd.DataFrame(columns=["anchor", "query_smiles", "row", "matched_smiles"])

    print(f"  canonicalizing {len(corpus_smiles)} corpus SMILES to locate "
          f"{len(wanted)} anchor(s)")
    corpus_canon = _canonicalize_many(list(corpus_smiles), n_workers)
    first_row: Dict[str, int] = {}
    for i, canon in enumerate(corpus_canon):
        if canon is not None and canon in wanted and canon not in first_row:
            first_row[canon] = i
    RDLogger.EnableLog("rdApp.*")

    rows = []
    for canon, entries in wanted.items():
        row = first_row.get(canon, -1)
        for name, query in entries:
            if row < 0:
                print(f"[warn] anchor {name!r} ({query}) is not in this corpus.")
            rows.append({"anchor": name, "query_smiles": query, "row": row,
                         "matched_smiles": corpus_smiles[row] if row >= 0 else ""})
    return pd.DataFrame(rows)


def cosine_neighbors(embedding: np.ndarray, query_rows: Sequence[int], k: int,
                     chunk_size: int = 200_000) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Top-`k` neighbours of each query row by cosine similarity.

    Cosine is the default throughout this pipeline (it is the UMAP metric for
    every continuous embedding), and it is what makes neighbours comparable
    across embeddings of different scale.

    Computed in chunks over the corpus so peak memory is chunk x n_queries
    rather than N x N. The query row itself is excluded from its own neighbours.
    """
    queries = np.asarray(list(query_rows), dtype=np.int64)
    if len(queries) == 0:
        return {}

    q = embedding[queries].astype(np.float32)
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)

    n = embedding.shape[0]
    k = int(min(k, n - 1))
    best_sim = np.full((len(queries), 0), 0.0, dtype=np.float32)
    best_idx = np.full((len(queries), 0), 0, dtype=np.int64)

    for start in tqdm(range(0, n, chunk_size), desc=f"Cosine kNN (k={k})"):
        end = min(start + chunk_size, n)
        block = embedding[start:end].astype(np.float32)
        norms = np.maximum(np.linalg.norm(block, axis=1, keepdims=True), 1e-12)
        sims = (block / norms) @ q.T                       # (chunk, n_queries)
        sims = sims.T                                      # (n_queries, chunk)

        idx = np.arange(start, end)[None, :].repeat(len(queries), axis=0)
        # Never let a query be its own neighbour.
        self_mask = idx == queries[:, None]
        sims = np.where(self_mask, -np.inf, sims)

        cat_sim = np.concatenate([best_sim, sims], axis=1)
        cat_idx = np.concatenate([best_idx, idx], axis=1)
        take = min(k, cat_sim.shape[1])
        part = np.argpartition(-cat_sim, take - 1, axis=1)[:, :take]
        rows = np.arange(len(queries))[:, None]
        best_sim, best_idx = cat_sim[rows, part], cat_idx[rows, part]

    order = np.argsort(-best_sim, axis=1)
    rows = np.arange(len(queries))[:, None]
    best_sim, best_idx = best_sim[rows, order], best_idx[rows, order]
    return {int(q_row): (best_idx[i], best_sim[i]) for i, q_row in enumerate(queries)}


@dataclass
class Region:
    """A 2-D window an inset zooms into."""
    region_id: str
    kind: str          # "anchor" | "cluster"
    label: str
    center_x: float
    center_y: float
    radius: float


def region_for_points(region_id: str, kind: str, label: str, xy: np.ndarray,
                      *, anchor_xy: Optional[np.ndarray] = None,
                      quantile: float = 0.9, padding_frac: float = 0.25,
                      min_radius: float = 1e-3, max_radius: float = np.inf) -> Region:
    """A window around `xy`, sized to be an actual zoom.

    Radius is a QUANTILE of the points' distance from the centre, not the
    maximum. That matters most for anchors: their cosine nearest neighbours live
    in the high-dimensional space and can land anywhere on the 2-D map, so a
    window covering all of them is the whole map and zooms into nothing. A
    quantile gives the local neighbourhood, and the dispersed neighbours remain
    visible on the main scatter where that dispersion is the interesting part.

    `anchor_xy` centres the window on the anchor itself rather than on the
    centroid of it plus its neighbours -- otherwise a few far-flung neighbours
    drag the window off the molecule it is supposed to be about.

    `min_radius`/`max_radius` keep a single-molecule region from becoming an
    empty box and a diffuse one from swallowing the figure; callers pass them
    relative to the map's own extent.
    """
    center = np.asarray(anchor_xy, dtype=float) if anchor_xy is not None else xy.mean(axis=0)
    if len(xy) <= 1:
        radius = min_radius
    else:
        dists = np.abs(xy - center).max(axis=1)
        radius = float(np.quantile(dists, quantile))
    radius = float(np.clip(radius * (1.0 + padding_frac), min_radius, max_radius))
    return Region(region_id, kind, label, float(center[0]), float(center[1]), radius)


def sample_region_molecules(rows: Sequence[int], smiles: Sequence[str], coords: np.ndarray,
                            n_mols: int, rng: np.random.Generator) -> List[Dict]:
    """Up to `n_mols` molecules from a region, for the inset's structure grid."""
    rows = np.asarray(list(rows), dtype=np.int64)
    if len(rows) == 0:
        return []
    take = rows if len(rows) <= n_mols else rng.choice(rows, size=n_mols, replace=False)
    return [{"row": int(r), "smiles": smiles[int(r)],
             "x": float(coords[int(r), 0]), "y": float(coords[int(r), 1])}
            for r in take]


def build_anchor_artifacts(
    *,
    embedding: np.ndarray,
    coords: np.ndarray,
    smiles: Sequence[str],
    anchor_specs: Sequence[Tuple[str, str]],
    k_neighbors: int,
    cluster_labels: Optional[np.ndarray],
    inset_clusters: Sequence[int],
    n_region_mols: int,
    seed: int,
    n_workers: int = 1,
) -> Dict[str, pd.DataFrame]:
    """Anchors, their cosine neighbours, inset regions, and region molecules."""
    rng = np.random.default_rng(seed)
    anchors = match_anchors(anchor_specs, smiles, n_workers=n_workers) \
        if anchor_specs else pd.DataFrame(columns=["anchor", "query_smiles", "row",
                                                   "matched_smiles"])

    if len(anchors):
        anchors["x"] = [float(coords[r, 0]) if r >= 0 else np.nan for r in anchors["row"]]
        anchors["y"] = [float(coords[r, 1]) if r >= 0 else np.nan for r in anchors["row"]]

    matched = anchors[anchors["row"] >= 0] if len(anchors) else anchors
    neighbor_rows: List[Dict] = []
    if len(matched) and k_neighbors > 0:
        found = cosine_neighbors(embedding, matched["row"].tolist(), k_neighbors)
        for _, a in matched.iterrows():
            idx, sim = found[int(a["row"])]
            for rank, (i, s) in enumerate(zip(idx, sim)):
                neighbor_rows.append({
                    "anchor": a["anchor"], "rank": rank, "row": int(i),
                    "cosine_sim": float(s), "smiles": smiles[int(i)],
                    "x": float(coords[int(i), 0]), "y": float(coords[int(i), 1]),
                })
    neighbors = pd.DataFrame(neighbor_rows, columns=["anchor", "rank", "row", "cosine_sim",
                                                     "smiles", "x", "y"])

    # --- regions -------------------------------------------------------------
    # Window sizes are bounded relative to the map's own extent: an inset must
    # be a zoom (not the whole map) and must not be an empty box around one
    # point.
    extent = float(np.abs(coords - coords.mean(axis=0)).max()) if len(coords) else 1.0
    min_radius, max_radius = 0.03 * extent, 0.35 * extent

    regions: List[Region] = []
    region_mols: List[Dict] = []

    for _, a in matched.iterrows():
        member_rows = [int(a["row"])]
        if len(neighbors):
            member_rows += neighbors[neighbors["anchor"] == a["anchor"]]["row"].tolist()
        region = region_for_points(
            f"anchor:{a['anchor']}", "anchor", str(a["anchor"]), coords[member_rows],
            anchor_xy=coords[int(a["row"])], quantile=0.5,
            min_radius=min_radius, max_radius=max_radius)
        regions.append(region)
        # The anchor is always the first molecule drawn in its own inset, then
        # its nearest neighbours in rank order -- an inset that sampled randomly
        # would not show what the anchor is actually near.
        for rank, r in enumerate(member_rows[:max(1, n_region_mols)]):
            region_mols.append({"region_id": region.region_id, "rank": rank, "row": int(r),
                                "smiles": smiles[int(r)],
                                "x": float(coords[int(r), 0]),
                                "y": float(coords[int(r), 1])})

    if cluster_labels is not None:
        for cid in inset_clusters:
            member_rows = np.flatnonzero(cluster_labels == cid)
            if len(member_rows) == 0:
                print(f"[warn] inset cluster {cid} has no members -- skipping.")
                continue
            region = region_for_points(
                f"cluster:{cid}", "cluster", f"cluster {cid}", coords[member_rows],
                quantile=0.9, min_radius=min_radius, max_radius=max_radius)
            regions.append(region)
            for rank, m in enumerate(sample_region_molecules(member_rows, smiles, coords,
                                                             n_region_mols, rng)):
                region_mols.append({"region_id": region.region_id, "rank": rank, **m})

    regions_df = pd.DataFrame([r.__dict__ for r in regions],
                              columns=["region_id", "kind", "label", "center_x", "center_y",
                                       "radius"])
    region_mols_df = pd.DataFrame(region_mols,
                                  columns=["region_id", "rank", "row", "smiles", "x", "y"])
    return {"anchors": anchors, "anchor_neighbors": neighbors,
            "regions": regions_df, "region_mols": region_mols_df}
