#!/usr/bin/env python
"""
05_umap.py
==========

General-purpose UMAP projection of any embedding, with every colour column a
figure might need stored alongside it.

Merges the previous pipeline's `5_run_pretty_plots_experiment.py`,
`6_run_pretty_plots_batch_experiment.py` and `7_run_umap_experiment.py`. Those
three re-ran the entire load-and-fit once per colouring, via a batch driver that
shelled out to itself. Here the projection is computed once per embedding and
every colouring is a plot-time choice.

Embeddings (`--embeddings`)
---------------------------
    global_cond        the peak embedder's pooled output (encoder)
    ecfp               Morgan fingerprint -- a model-free structural baseline
    x:5@0.001          decoder atom/coord stream, layer 5, timestep 0.001
    y:11@1.0           decoder NMR stream, layer 11, timestep 1.0
    x:5                ... first timestep available for that layer
    all-decoder        every (stream, layer, timestep) in the layerwise file
    all                global_cond + ecfp + all-decoder

Each gets its own tag, so `data/05_umap/global_cond/`,
`data/05_umap/ecfp/`, `data/05_umap/decoder_x_L05_t0.001/`, ...

Reusing existing fits
---------------------
UMAP over millions of molecules costs hours, so fits are cached by a
hyperparameter fingerprint. `--reuse-cache-dir` points at the PREVIOUS
pipeline's `umap_shared_cache/`, which already holds full-corpus (2,524,941
molecule) fits for `global_cond` (cosine, scaled) and `ecfp` (jaccard,
unscaled) at the default hyperparameters. Those are found and loaded rather
than refit, as long as the corpus and hyperparameters match exactly. New fits
are written to `--umap-cache-dir` (this repo's `cache/umap/`), never to the
read-only one.

The fingerprint is a CONFIG hash, not a content hash: same shape + same
hyperparameters matches, even for different data. Use a separate cache dir per
corpus if that matters.

Colour columns written (all aligned to the projection's rows)
-------------------------------------------------------------
    coords.npz        the 2-D projection
    categorical.npz   dataset / split / real-vs-synthetic / Butina cluster
    properties.npz    RDKit molecular properties (--properties)
    spectral.npz      the NMR spectral feature panel from extraction/01

Usage
-----
    python analysis/05_umap.py \
        --data-dir /scratch/.../epoch1399 --prefix cotrain --splits train \
        --embeddings global_cond ecfp --properties extended
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402

from src.analysis.corpus import (  # noqa: E402
    available_decoder_keys,
    load_decoder_corpus,
    load_encoder_corpus,
    parse_embedding_specs,
)
from src.analysis.anchors import parse_anchor_specs, read_anchor_file  # noqa: E402
from src.analysis.properties import compute_properties, resolve_property_names  # noqa: E402
from src.analysis.umap_cache import (  # noqa: E402
    DEFAULT_PCA_DIM,
    DEFAULT_UMAP_MIN_DIST,
    DEFAULT_UMAP_N_NEIGHBORS,
    DEFAULT_UMAP_SEED,
)
from src.analysis.umap_projection import load_cluster_labels, project_and_store  # noqa: E402
from src.common.paths import cache_dir, data_dir  # noqa: E402
from src.common.workers import safe_n_workers  # noqa: E402

SLUG = "05_umap"

# The previous pipeline's cache, which already holds full-corpus global_cond and
# ecfp fits. Read-only: new fits go to this repo's own cache.
PREVIOUS_PIPELINE_CACHE = (
    _REPO_ROOT.parent / "8_11_26_cotrainv3_dedupOFF_s0_epoch1399" / "umap_shared_cache"
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Project embeddings to 2-D with UMAP and store every colour column.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--layerwise-dir", type=Path, default=None,
                   help="Default: --data-dir.")
    p.add_argument("--spectral-dir", type=Path, default=None,
                   help="Default: --data-dir.")
    p.add_argument("--prefix", type=str, default="cotrain")
    p.add_argument("--splits", nargs="+", default=["train"], choices=["train", "val", "test"])

    p.add_argument("--embeddings", nargs="+", default=["global_cond"],
                   help="'global_cond', 'ecfp', 'all', 'all-decoder', or decoder specs "
                        "like 'x:5@0.001'.")
    p.add_argument("--list-embeddings", action="store_true",
                   help="Print the decoder embeddings available, then exit.")
    p.add_argument("--tag-suffix", type=str, default=None,
                   help="Appended to each embedding's tag, for parameter variants.")

    g = p.add_argument_group("UMAP")
    g.add_argument("--umap-n-neighbors", type=int, default=DEFAULT_UMAP_N_NEIGHBORS)
    g.add_argument("--umap-min-dist", type=float, default=DEFAULT_UMAP_MIN_DIST)
    g.add_argument("--umap-metric", type=str, default=None,
                   help="Default: cosine for continuous embeddings, jaccard for ECFP.")
    g.add_argument("--pca-dim", type=int, default=DEFAULT_PCA_DIM,
                   help="PCA dimensions before UMAP. 0 disables the PCA step.")
    g.add_argument("--scale", dest="scale", action="store_true", default=None,
                   help="Force StandardScaler before PCA/UMAP.")
    g.add_argument("--no-scale", dest="scale", action="store_false",
                   help="Force no scaling. Default: on for continuous, off for ECFP.")
    g.add_argument("--umap-seed", type=int, default=DEFAULT_UMAP_SEED)
    g.add_argument("--umap-cache-dir", type=Path, default=None,
                   help="Where new fits are cached. Default: cache/umap/.")
    g.add_argument("--reuse-cache-dir", type=Path, default=PREVIOUS_PIPELINE_CACHE,
                   help="Read-only cache searched on a miss. Defaults to the previous "
                        "pipeline's umap_shared_cache/, which holds full-corpus "
                        "global_cond and ecfp fits.")
    g.add_argument("--no-reuse-cache", action="store_true",
                   help="Ignore the read-only cache entirely.")

    c = p.add_argument_group("colour columns")
    c.add_argument("--properties", nargs="+", default=["basic"],
                   help="'basic' (5), 'extended' (~34), 'all' (every RDKit descriptor), "
                        "or explicit names.")
    c.add_argument("--n-property-workers", type=int, default=safe_n_workers())
    c.add_argument("--cluster-tag", type=str, default="main",
                   help="Which analysis/03_clustering run supplies cluster ids, i.e. "
                        "data/03_clustering/<tag>/. Pass '' to skip.")
    c.add_argument("--real-prefixes", nargs="+", default=["real"],
                   help="dataset-name prefixes counted as 'real'.")
    c.add_argument("--sim-prefixes", nargs="+", default=["syn"],
                   help="dataset-name prefixes counted as 'synthetic'.")

    a = p.add_argument_group("anchors, neighbours and insets")
    a.add_argument("--highlight-smiles", nargs="+", default=None,
                   help="Molecules to mark on the map. 'name=SMILES' or a bare SMILES. "
                        "Matched to the corpus by CANONICAL SMILES, so the string need not "
                        "be written the same way.")
    a.add_argument("--highlight-smiles-file", type=Path, default=None,
                   help="File with one 'name=SMILES' (or bare SMILES) per line; "
                        "'#' comments ignored.")
    a.add_argument("--knn", type=int, default=0,
                   help="Nearest neighbours to find per anchor, by COSINE similarity in "
                        "the high-dimensional embedding -- not in the 2-D map. 0 skips.")
    a.add_argument("--inset-clusters", nargs="+", type=int, default=None,
                   help="Butina cluster ids to build inset regions for, alongside any "
                        "anchors.")
    a.add_argument("--inset-top-clusters", type=int, default=0,
                   help="Instead of naming ids, build inset regions for the N largest "
                        "clusters.")
    a.add_argument("--n-region-mols", type=int, default=6,
                   help="Representative molecules stored per inset region.")
    return p.parse_args()


def main():
    args = parse_args()
    layerwise_dir = args.layerwise_dir or args.data_dir
    spectral_dir = args.spectral_dir or args.data_dir

    if args.list_embeddings:
        print("global_cond\necfp")
        for stream, layer, timestep in available_decoder_keys(layerwise_dir, args.prefix,
                                                              args.splits):
            print(f"{'x' if stream == 'x_hidden_mean' else 'y'}:{layer}@{timestep:g}")
        return

    specs = parse_embedding_specs(args.embeddings, layerwise_dir, args.prefix, args.splits)
    corpus_specs = [s for s in specs if s.kind in ("encoder", "ecfp")]
    decoder_specs = [s for s in specs if s.kind == "decoder"]

    umap_cache = args.umap_cache_dir or cache_dir("umap", create=True)
    reuse = [] if args.no_reuse_cache else [args.reuse_cache_dir]
    if reuse and not Path(args.reuse_cache_dir).exists():
        print(f"[note] read-only cache {args.reuse_cache_dir} does not exist; ignoring.")
        reuse = []

    print("=" * 78)
    print("UMAP projection")
    print(f"  data dir      : {args.data_dir}")
    print(f"  prefix/splits : {args.prefix} / {args.splits}")
    print(f"  embeddings    : {len(specs)} -> {[s.slug for s in specs]}")
    print(f"  umap cache    : {umap_cache}")
    print(f"  reuse cache   : {reuse[0] if reuse else '(none)'}")
    print("=" * 78)

    property_names = resolve_property_names(args.properties)
    cluster_dir = data_dir("03_clustering", args.cluster_tag) if args.cluster_tag else None
    pca_dim = args.pca_dim if args.pca_dim and args.pca_dim > 0 else None

    anchor_specs = list(parse_anchor_specs(args.highlight_smiles or []))
    if args.highlight_smiles_file:
        anchor_specs += read_anchor_file(args.highlight_smiles_file)
    if anchor_specs:
        print(f"  anchors: {', '.join(name for name, _ in anchor_specs)}"
              + (f" (+{args.knn} cosine NN each)" if args.knn else ""))

    inputs = [Path(args.data_dir) / f"{args.prefix}_{s}_global_cond.pt" for s in args.splits]
    inputs += [Path(spectral_dir) / f"{args.prefix}_{s}_spectral_features.pt"
               for s in args.splits]

    def tag_for(spec) -> str:
        return f"{spec.slug}_{args.tag_suffix}" if args.tag_suffix else spec.slug

    def resolve_inset_clusters(clusters) -> list:
        """Explicit ids, or the N largest, or nothing."""
        if args.inset_clusters:
            return list(args.inset_clusters)
        if args.inset_top_clusters and clusters is not None:
            ids, counts = np.unique(clusters, return_counts=True)
            return ids[np.argsort(-counts)][:args.inset_top_clusters].tolist()
        return []

    def store(spec, embedding, smiles, dataset, split, spectral, spectral_names,
              properties, clusters):
        tag = tag_for(spec)
        print(f"\n### {spec.label} -> data/{SLUG}/{tag}/ ###")
        project_and_store(
            embedding=embedding, smiles=smiles, dataset=dataset, split=split,
            properties=properties, property_names=property_names,
            spectral=spectral, spectral_names=spectral_names,
            cluster_labels=clusters,
            anchor_specs=anchor_specs, k_neighbors=args.knn,
            inset_clusters=resolve_inset_clusters(clusters),
            n_region_mols=args.n_region_mols, seed=args.umap_seed,
            n_workers=args.n_property_workers,
            out_dir=data_dir(SLUG, tag, create=True), tag=tag, slug=SLUG,
            embedding_key=spec.cache_key, embedding_label=spec.label,
            n_neighbors=args.umap_n_neighbors, min_dist=args.umap_min_dist,
            metric=args.umap_metric, pca_dim=pca_dim, scale=args.scale,
            umap_seed=args.umap_seed, cache_dir=umap_cache,
            read_only_cache_dirs=reuse,
            real_prefixes=args.real_prefixes, sim_prefixes=args.sim_prefixes,
            params={"prefix": args.prefix, "splits": list(args.splits),
                    "data_dir": str(args.data_dir), "cluster_tag": args.cluster_tag},
            inputs=inputs,
        )

    # --- encoder / ecfp -------------------------------------------------------
    if corpus_specs:
        keys = [("global_cond" if s.kind == "encoder" else "ecfp") for s in corpus_specs]
        print(f"\n### loading the corpus ({', '.join(keys)}) ###")
        corpus = load_encoder_corpus(args.data_dir, spectral_dir, args.prefix, args.splits,
                                     embedding_keys=tuple(dict.fromkeys(keys)),
                                     require_spectral=False)
        n = len(corpus["dataset"])
        print(f"Corpus: {n} molecules")
        properties = compute_properties(corpus["smiles"], property_names,
                                        args.n_property_workers)
        clusters = load_cluster_labels(cluster_dir, n)
        for spec in corpus_specs:
            key = "global_cond" if spec.kind == "encoder" else "ecfp"
            store(spec, corpus["embeddings"][key], corpus["smiles"], corpus["dataset"],
                  corpus["split"], corpus["spectral_features"],
                  corpus["spectral_feature_names"], properties, clusters)
        del corpus

    # --- decoder --------------------------------------------------------------
    if decoder_specs:
        print("\n### loading the decoder corpus ###")
        corpus = load_decoder_corpus(args.data_dir, layerwise_dir, spectral_dir, args.prefix,
                                     args.splits, decoder_specs,
                                     n_workers=args.n_property_workers)
        n = len(corpus["smiles"])
        print(f"Corpus: {n} molecules matched to the cotrain corpus")
        properties = compute_properties(corpus["smiles"], property_names,
                                        args.n_property_workers)
        # Cluster labels are indexed in the FULL corpus's row order, so a decoder
        # run gathers them through the same join its spectra came through rather
        # than reading them positionally.
        clusters = None
        if cluster_dir is not None:
            path = Path(cluster_dir) / "cluster_labels.npy"
            if not path.exists():
                print(f"[note] No cluster labels at {path} -- 'cluster' colouring will be "
                      f"unavailable.")
            else:
                all_labels = np.load(path)
                ref = corpus["peak_index"]
                if len(all_labels) <= int(ref.max()):
                    print(f"[warn] cluster labels cover {len(all_labels)} molecules but the "
                          f"join reaches row {int(ref.max())} -- the clustering was run over "
                          f"different --splits. Skipping 'cluster' colouring.")
                else:
                    clusters = all_labels[ref].astype(np.int32)
                    print(f"  cluster labels gathered through the decoder join "
                          f"({len(np.unique(clusters))} clusters)")

        for spec in decoder_specs:
            store(spec, corpus["embeddings"][(spec.stream, spec.layer, spec.timestep)],
                  corpus["smiles"], corpus["dataset"],
                  np.asarray(["+".join(args.splits)] * n),
                  corpus["spectral_features"], corpus["spectral_feature_names"],
                  properties, clusters)

    print(f"\nDone. Draw figures with:\n"
          f"    python plotting/{SLUG}.py --tag {tag_for(specs[0])} --color-by dataset")


if __name__ == "__main__":
    main()
