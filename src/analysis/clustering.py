"""Butina clustering of the corpus over ECFP fingerprints.

Butina is exact and quadratic, so it cannot be run over millions of molecules
directly. The strategy, unchanged from the previous pipeline:

  1. take a **stratified** subsample (proportional per source dataset) small
     enough to cluster exactly;
  2. run Butina on it, and keep each cluster's first member -- Butina's own
     centroid -- as that cluster's prototype fingerprint;
  3. assign every molecule in the full corpus to its nearest prototype by
     Tanimoto similarity, in chunks;
  4. overwrite the subsample's rows with their exact Butina labels, so the
     molecules that were actually clustered keep their true assignment rather
     than a nearest-prototype approximation of it.

Step 3 is what makes this scale, and step 4 is what keeps it honest.

Computation only -- never imports matplotlib.
"""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.analysis.corpus import cached_descriptor_matrix
from src.common.manifest import write_manifest
from src.common.paths import cache_dir


SCHEMA_VERSION = 1

# The four descriptors the cluster-summary bars show. RDKit's own names are on
# the left; the display names on the right are what the previous pipeline's
# figure used, and are kept so the bars stay labelled the same way.
CLUSTER_DESCRIPTORS = ["MolWt", "MolLogP", "TPSA", "RingCount"]
PROPS = ["MolWt", "LogP", "TPSA", "Rings"]


def load_fingerprints(data_dir: Path, prefix: str, splits: Sequence[str]) -> Dict:
    """SMILES, source dataset, split tag and ECFP for the whole corpus."""
    smiles: List[str] = []
    dataset: List[str] = []
    split_tag: List[str] = []
    ecfp_parts: List[torch.Tensor] = []
    ecfp_radius = ecfp_nbits = None

    for split in splits:
        path = Path(data_dir) / f"{prefix}_{split}_global_cond.pt"
        if not path.exists():
            raise SystemExit(f"Expected extraction output not found: {path}\n"
                             f"Run extraction/00_global_cond.py first.")
        d = torch.load(path, map_location="cpu", weights_only=False)
        if ecfp_radius is None:
            ecfp_radius, ecfp_nbits = d["ecfp_radius"], d["ecfp_nbits"]
        elif (ecfp_radius, ecfp_nbits) != (d["ecfp_radius"], d["ecfp_nbits"]):
            # Fingerprints of different radius/width are not comparable, so a
            # Tanimoto between them is meaningless rather than merely noisy.
            raise SystemExit(
                f"Mismatched ECFP settings across splits: {path.name} has "
                f"radius={d['ecfp_radius']}, nBits={d['ecfp_nbits']} but an earlier split had "
                f"radius={ecfp_radius}, nBits={ecfp_nbits}. Re-extract them consistently.")

        smiles.extend(d["smiles"])
        dataset.extend(d["dataset"])
        split_tag.extend([split] * len(d["smiles"]))
        ecfp_parts.append(d["ecfp"])

    return {
        "smiles": smiles,
        "dataset": np.asarray(dataset),
        "split": np.asarray(split_tag),
        "ecfp": torch.cat(ecfp_parts, dim=0).numpy().astype(np.float32),
        "ecfp_radius": ecfp_radius,
        "ecfp_nbits": ecfp_nbits,
    }


def stratified_subsample_indices(dataset_labels: np.ndarray, n_sample: int,
                                 rng: np.random.Generator) -> np.ndarray:
    """Indices for a subsample that keeps each source dataset's share.

    Stratifying matters because Butina runs only on this subsample: a plain
    uniform draw would let the largest source dominate the discovered clusters,
    and every smaller source would then be described by prototypes that were
    never fit to it.
    """
    dataset_labels = np.asarray(dataset_labels)
    n_total = len(dataset_labels)
    n_sample = min(n_sample, n_total)

    picked = []
    for src in sorted(set(dataset_labels.tolist())):
        src_idx = np.where(dataset_labels == src)[0]
        n_take = min(max(1, round(n_sample * len(src_idx) / n_total)), len(src_idx))
        picked.append(rng.choice(src_idx, size=n_take, replace=False))

    idx = np.concatenate(picked)
    if len(idx) > n_sample:
        idx = rng.choice(idx, size=n_sample, replace=False)
    rng.shuffle(idx)
    return idx


# Butina needs the whole lower triangle in memory as a Python float list, which
# costs roughly 32 bytes per pair. At 10k molecules that is ~1.6 GB; at 100k it
# would be ~160 GB. The subsample size is the one knob that can quietly turn this
# analysis into an hours-long run that then dies, so it is checked up front.
_MAX_CONDENSED_PAIRS = 200_000_000  # ~6.4 GB of Python floats


def _check_subsample_size(n: int) -> None:
    n_pairs = n * (n - 1) // 2
    if n_pairs > _MAX_CONDENSED_PAIRS:
        raise SystemExit(
            f"--n-cluster-sample {n} means {n_pairs:,} pairwise distances, which Butina "
            f"needs held in memory at once (~{n_pairs * 32 / 2**30:.0f} GB as Python "
            f"floats).\n"
            f"Butina is exact and quadratic; it is meant to run on a subsample, with the "
            f"full corpus reached by the nearest-prototype assignment pass that follows.\n"
            f"Use --n-cluster-sample 10000-20000 (the previous pipeline's default was "
            f"10000, giving a few thousand clusters over this corpus).")


def tanimoto_condensed_distances(ecfp_sub: np.ndarray, device: str) -> List[float]:
    """Condensed (lower-triangle) Tanimoto distance list, as Butina wants it."""
    _check_subsample_size(ecfp_sub.shape[0])
    x = torch.from_numpy(ecfp_sub).to(device)
    row_sums = x.sum(dim=1)

    dists: List[float] = []
    for i in tqdm(range(1, x.shape[0]), desc="Pairwise Tanimoto (subsample)"):
        inter = (x[i:i + 1] @ x[:i].T).squeeze(0)
        union = row_sums[i] + row_sums[:i] - inter
        sim = torch.where(union > 0, inter / union, torch.zeros_like(union))
        dists.extend((1.0 - sim).cpu().tolist())
    return dists


def run_butina(ecfp_sub: np.ndarray, cutoff: float, device: str):
    from rdkit.ML.Cluster import Butina

    dists = tanimoto_condensed_distances(ecfp_sub, device)
    return Butina.ClusterData(dists, ecfp_sub.shape[0], cutoff, isDistData=True)


# Peak intermediate size for the assignment pass, in float32 elements. Three
# (chunk x n_prototypes) tensors are live at once, so this is ~768 MB at 64M.
_ASSIGN_ELEMENT_BUDGET = 64_000_000


def _assign_chunk(x_np: np.ndarray, protos: torch.Tensor, proto_sums: torch.Tensor,
                  device: str) -> np.ndarray:
    x = torch.from_numpy(x_np).to(device)
    x_sums = x.sum(dim=1, keepdim=True)
    inter = x @ protos.T
    union = x_sums + proto_sums.unsqueeze(0) - inter
    sim = torch.where(union > 0, inter / union, torch.zeros_like(union))
    return sim.argmax(dim=1).cpu().numpy().astype(np.int32)


def assign_to_prototypes(ecfp_all: np.ndarray, prototype_fps: np.ndarray, device: str,
                         chunk_size: int = 50_000) -> np.ndarray:
    """Nearest prototype by Tanimoto, for every molecule, in chunks.

    `chunk_size` is a CAP, not the operative value. Memory here scales with
    chunk x n_prototypes, not with chunk alone, and Butina routinely yields
    thousands of prototypes -- so a chunk size that is fine for one clustering
    can OOM on another with the same corpus. The effective chunk is derived from
    the prototype count, then halved on any OOM, and finally falls back to CPU
    rather than losing the run. (The previous pipeline used the raw chunk size
    and simply died on a busy GPU.)
    """
    n, n_protos = ecfp_all.shape[0], prototype_fps.shape[0]
    effective = max(1, min(chunk_size, _ASSIGN_ELEMENT_BUDGET // max(n_protos, 1)))
    if effective < chunk_size:
        print(f"  assignment chunk {chunk_size} -> {effective} "
              f"({n_protos} prototypes; keeps intermediates near "
              f"{effective * n_protos * 4 / 2**20:.0f} MB each)")

    protos = torch.from_numpy(prototype_fps).to(device)
    proto_sums = protos.sum(dim=1)
    labels = np.empty(n, dtype=np.int32)

    start = 0
    with tqdm(total=n, desc="Assigning molecules to clusters") as bar:
        while start < n:
            end = min(start + effective, n)
            try:
                labels[start:end] = _assign_chunk(ecfp_all[start:end], protos, proto_sums,
                                                  device)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if effective > 256:
                    effective = max(256, effective // 2)
                    print(f"\n[warn] CUDA OOM -- retrying with chunk {effective}. "
                          f"(This GPU is shared; other processes may be using it.)")
                    continue
                print("\n[warn] CUDA OOM even at the minimum chunk size -- finishing this "
                      "pass on CPU. It is slower but produces identical labels.")
                device = "cpu"
                protos = protos.cpu()
                proto_sums = proto_sums.cpu()
                effective = min(chunk_size, _ASSIGN_ELEMENT_BUDGET // max(n_protos, 1))
                continue
            bar.update(end - start)
            start = end
    return labels


def _prototype_cache_path(n_molecules: int, n_cluster_sample: int, cutoff: float,
                          seed: int, ecfp_radius: int, ecfp_nbits: int,
                          dataset: np.ndarray) -> Path:
    """Keyed on everything the Butina step depends on.

    The source-dataset composition is folded in because the subsample is
    stratified by it -- two corpora of the same size with different mixtures
    would otherwise collide on one cache entry.
    """
    names, counts = np.unique(dataset, return_counts=True)
    key = hashlib.sha256(
        f"{n_molecules}|{n_cluster_sample}|{cutoff}|{seed}|{ecfp_radius}|{ecfp_nbits}|"
        f"{list(zip(names.tolist(), counts.tolist()))}".encode()
    ).hexdigest()[:32]
    return cache_dir("butina", create=True) / f"{key}.pkl"


def run_clustering(
    *,
    data_dir: Path,
    prefix: str,
    splits: Sequence[str],
    out_dir: Path,
    tag: str,
    slug: str,
    n_cluster_sample: int = 10_000,
    butina_cutoff: float = 0.35,
    max_clusters_plot: int = 24,
    n_reps: int = 6,
    assign_chunk_size: int = 50_000,
    n_desc_workers: int = 1,
    device: str = "cpu",
    seed: int = 1234,
    refit: bool = False,
    load_prototypes: Optional[Path] = None,
    write_meta_csv: bool = False,
    use_descriptor_cache: bool = True,
) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    data = load_fingerprints(data_dir, prefix, splits)
    smiles, ecfp = data["smiles"], data["ecfp"]
    n_total = len(smiles)
    print(f"Loaded {n_total} molecules (ECFP radius={data['ecfp_radius']}, "
          f"nBits={data['ecfp_nbits']})")

    # --- Butina on the stratified subsample (cached) --------------------------
    cache_path = load_prototypes or _prototype_cache_path(
        n_total, n_cluster_sample, butina_cutoff, seed,
        data["ecfp_radius"], data["ecfp_nbits"], data["dataset"])

    if cache_path.exists() and not refit:
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        sub_idx = cache["sub_idx"]
        sub_labels_local = cache["sub_labels_local"]
        prototype_fps = cache["prototype_fps"]
        n_clusters = int(cache["n_clusters"])
        print(f"Reusing cached Butina prototypes {cache_path.name} "
              f"(n_clusters={n_clusters}, cutoff={cache['butina_cutoff']}, "
              f"n_cluster_sample={cache['n_cluster_sample']}). Pass --refit to redo it.")
    else:
        sub_idx = stratified_subsample_indices(data["dataset"], n_cluster_sample, rng)
        ecfp_sub = ecfp[sub_idx]
        print(f"Subsampled {len(sub_idx)} molecules for exact clustering "
              f"(stratified by source dataset).")
        clusters = run_butina(ecfp_sub, butina_cutoff, device)
        print(f"Butina produced {len(clusters)} clusters on the subsample "
              f"(cutoff={butina_cutoff}).")

        sub_labels_local = np.full(len(sub_idx), -1, dtype=np.int32)
        prototypes = []
        for cluster_id, members in enumerate(clusters):
            sub_labels_local[list(members)] = cluster_id
            prototypes.append(ecfp_sub[members[0]])  # Butina's own centroid
        prototype_fps = np.stack(prototypes, axis=0)
        n_clusters = len(clusters)

        with open(cache_path, "wb") as f:
            pickle.dump({"sub_idx": sub_idx, "sub_labels_local": sub_labels_local,
                         "prototype_fps": prototype_fps, "n_clusters": n_clusters,
                         "butina_cutoff": butina_cutoff,
                         "n_cluster_sample": n_cluster_sample, "seed": seed,
                         "ecfp_radius": data["ecfp_radius"],
                         "ecfp_nbits": data["ecfp_nbits"]}, f)
        print(f"Cached Butina prototypes -> {cache_path}")

    # --- assign the full corpus ----------------------------------------------
    labels = assign_to_prototypes(ecfp, prototype_fps, device, assign_chunk_size)
    # The clustered molecules keep their EXACT Butina label rather than a
    # nearest-prototype approximation of it.
    labels[sub_idx] = sub_labels_local

    written["cluster_labels"] = out_dir / "cluster_labels.npy"
    np.save(written["cluster_labels"], labels)
    print(f"Saved cluster labels (n_clusters={n_clusters})")

    # --- descriptors + per-cluster summary ------------------------------------
    desc = cached_descriptor_matrix(smiles, CLUSTER_DESCRIPTORS, n_desc_workers,
                                    use_cache=use_descriptor_cache)
    desc_df = pd.DataFrame(desc, columns=PROPS)

    written["descriptors"] = out_dir / "descriptors.npz"
    np.savez_compressed(written["descriptors"], values=desc.astype(np.float32),
                        names=np.asarray(PROPS))

    unique_clusters, counts = np.unique(labels, return_counts=True)
    order = np.argsort(-counts)
    rows = []
    for cluster in unique_clusters[order]:
        idx = np.where(labels == cluster)[0]
        row = {"cluster": int(cluster), "n_molecules": len(idx)}
        row.update(desc_df.iloc[idx].mean(numeric_only=True).to_dict())
        rows.append(row)
    stats_df = pd.DataFrame(rows)
    written["cluster_stats"] = out_dir / "cluster_stats.csv"
    stats_df.to_csv(written["cluster_stats"], index=False)
    print(f"Saved per-cluster stats ({len(stats_df)} clusters)")

    # --- what the figure draws ------------------------------------------------
    plot_df = stats_df.head(max_clusters_plot).reset_index(drop=True)
    # Bars are min-maxed across the PLOTTED clusters, so a bar reads as "high
    # for this panel of clusters". Done here rather than at draw time: it is a
    # computation over the data, and the figure should not have to re-derive it.
    scaled = plot_df[PROPS].copy()
    span = (scaled.max() - scaled.min()).replace(0, 1)
    for prop in PROPS:
        plot_df[f"{prop}_norm"] = ((scaled[prop] - scaled[prop].min()) / span[prop]).values
    written["plot_clusters"] = out_dir / "plot_clusters.csv"
    plot_df.to_csv(written["plot_clusters"], index=False)

    rep_rows = []
    for cluster in plot_df["cluster"]:
        idx = np.where(labels == int(cluster))[0]
        # NOTE: the previous pipeline rebound `n_reps` inside this loop, so the
        # first small cluster permanently shrank the count for every cluster
        # after it. Kept local here.
        take = min(n_reps, len(idx))
        for rank, i in enumerate(rng.choice(idx, size=take, replace=False)):
            rep_rows.append({"cluster": int(cluster), "rank": rank, "smiles": smiles[i]})
    written["representatives"] = out_dir / "representatives.csv"
    pd.DataFrame(rep_rows).to_csv(written["representatives"], index=False)

    # --- per-molecule table, opt-in -------------------------------------------
    # The old pipeline always wrote this. It is not figure data, and at full
    # corpus scale it is a ~500 MB CSV whose only new columns are `cluster` and
    # the four descriptors -- both already saved compactly above, aligned to the
    # extraction's row order. Kept behind a flag for when the flat table is
    # genuinely wanted.
    if write_meta_csv:
        meta = pd.DataFrame({"smiles": smiles, "dataset": data["dataset"],
                             "split": data["split"], "cluster": labels})
        written["cluster_meta"] = out_dir / "cluster_meta.csv"
        pd.concat([meta, desc_df], axis=1).to_csv(written["cluster_meta"], index=False)
        print(f"Saved per-molecule meta CSV")

    dataset_counts = {str(k): int(v) for k, v in zip(*np.unique(data["dataset"],
                                                                return_counts=True))}
    write_manifest(
        out_dir, slug=slug, tag=tag, schema_version=SCHEMA_VERSION,
        params={
            "prefix": prefix, "splits": list(splits), "data_dir": str(data_dir),
            "n_molecules": n_total,
            "n_clusters": n_clusters,
            "n_cluster_sample": int(n_cluster_sample),
            "n_subsampled": int(len(sub_idx)),
            "butina_cutoff": butina_cutoff,
            "max_clusters_plot": max_clusters_plot,
            "n_reps": n_reps,
            "seed": seed,
            "device": device,
            "ecfp_radius": data["ecfp_radius"],
            "ecfp_nbits": data["ecfp_nbits"],
            "props": PROPS,
            "prototype_cache": str(cache_path),
            "largest_clusters": [{"cluster": int(r["cluster"]),
                                  "n_molecules": int(r["n_molecules"])}
                                 for _, r in stats_df.head(5).iterrows()],
            "dataset_counts": dataset_counts,
        },
        inputs=[Path(data_dir) / f"{prefix}_{s}_global_cond.pt" for s in splits],
        outputs=list(written.values()),
    )
    return written
