#!/usr/bin/env python
"""
04_joint_pca.py
===============

Joint structural + spectral PCA over a chosen embedding space.

Fits ONE PCA (StandardScaler + PCA) and explains its components in two
vocabularies at once -- RDKit structural descriptors, and NMR spectral features
-- so the structure filmstrip and the spectrum filmstrip are guaranteed views of
the same axis. Port of the previous pipeline's `3_run_joint_pca_experiment.py`,
which merged its scripts `3` and `17`.

What is new here: the embedding is **selectable**. The old script always ran over
`global_cond`; this one runs over the encoder embedding, any decoder trunk hidden
state, or all of them in one invocation.

    --embeddings global_cond          the peak embedder's pooled output (encoder)
    --embeddings x:5@0.001            decoder atom/coord stream, layer 5, t=0.001
    --embeddings y:11@1.0             decoder NMR stream, layer 11, t=1.0
    --embeddings x:5                  ... first timestep available for that layer
    --embeddings all-decoder          every (stream, layer, timestep) in the file
    --embeddings all                  global_cond + all-decoder

Each embedding gets its own output tag, so a multi-embedding run writes
`data/04_joint_pca/global_cond/`, `data/04_joint_pca/decoder_x_L05_t0.001/`, and
so on. `--list-embeddings` prints what the layerwise file actually contains.

Decoder embeddings are joined to the corpus
-------------------------------------------
`extraction/02_decoder_layers.py` subsamples (`--n-samples-per-dataset`), so its
molecules are a subset of the `global_cond` corpus in a different order. SMILES,
dataset labels, spectral features and peak lists are all indexed through that
join; unmatched rows are dropped rather than carried as NaN. The join prefers
`mol_idx`, falls back to raw SMILES, then to canonical SMILES for whatever is
left.

Runs entirely from cached extraction output -- no checkpoint, no GPU.

Usage
-----
    python analysis/04_joint_pca.py \
        --data-dir /scratch/.../26-07-27-cotrain-v3-dedupOFF-s0/epoch1399 \
        --prefix cotrain --splits train --embeddings all --n-pcs 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.analysis.corpus import (  # noqa: E402
    available_decoder_keys,
    cached_descriptor_matrix,
    load_decoder_corpus,
    load_encoder_corpus,
    parse_embedding_specs,
)
from src.analysis.descriptors import DEFAULT_DESCRIPTOR_NAMES  # noqa: E402
from src.analysis.joint_pca import fit_joint_pca  # noqa: E402
from src.common.paths import data_dir  # noqa: E402
from src.common.workers import safe_n_workers  # noqa: E402

SLUG = "04_joint_pca"


def parse_args():
    p = argparse.ArgumentParser(
        description="Joint structural + spectral PCA over encoder and/or decoder embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--data-dir", type=Path, required=True,
                   help="Directory holding extraction's <prefix>_<split>_global_cond.pt.")
    p.add_argument("--layerwise-dir", type=Path, default=None,
                   help="Directory holding <prefix>_<split>_layerwise.pt. Default: --data-dir.")
    p.add_argument("--spectral-dir", type=Path, default=None,
                   help="Directory holding <prefix>_<split>_spectral_features.pt. "
                        "Default: --data-dir.")
    p.add_argument("--prefix", type=str, default="cotrain")
    p.add_argument("--splits", nargs="+", default=["train"], choices=["train", "val", "test"])

    p.add_argument("--embeddings", nargs="+", default=["global_cond"],
                   help="Which embedding spaces to analyze. 'global_cond', 'all', "
                        "'all-decoder', or decoder specs like 'x:5@0.001'.")
    p.add_argument("--list-embeddings", action="store_true",
                   help="Print the decoder embeddings available in the layerwise file, "
                        "then exit.")
    p.add_argument("--tag-suffix", type=str, default=None,
                   help="Appended to each embedding's tag, to keep parameter variants side "
                        "by side (e.g. --n-pcs 16 --tag-suffix npcs16).")

    p.add_argument("--n-pcs", type=int, default=8)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--no-scale", action="store_true",
                   help="Skip the StandardScaler before PCA. The previous pipeline scaled by "
                        "default; this keeps that default.")
    p.add_argument("--descriptor-names", nargs="+", default=None,
                   help=f"Override the RDKit descriptor panel. Default: "
                        f"{len(DEFAULT_DESCRIPTOR_NAMES)} built-ins.")
    p.add_argument("--n-desc-workers", type=int, default=safe_n_workers(),
                   help="Workers for the RDKit descriptor pass -- the dominant cost of a "
                        "full-corpus run.")
    p.add_argument("--no-descriptor-cache", action="store_true",
                   help="Recompute descriptors instead of reusing cache/descriptors/. The "
                        "cache is keyed on the exact SMILES list, so this is only useful "
                        "when debugging the descriptor pass itself.")
    p.add_argument("--top-k", type=int, default=5,
                   help="Rows per PC in pc_top_spectral_features.csv.")

    # Traversal geometry. These decide WHICH molecules the filmstrip shows, so
    # they are analysis parameters, not cosmetics -- unlike in the old script,
    # where they were passed at plot time.
    p.add_argument("--n-steps", type=int, default=8,
                   help="Steps per PC traversal.")
    p.add_argument("--pc-percentile-lo", type=float, default=1.0,
                   help="Lower percentile of each PC used as the traversal's start.")
    p.add_argument("--pc-percentile-hi", type=float, default=99.0,
                   help="Upper percentile of each PC used as the traversal's end.")
    p.add_argument("--n-bar-features", type=int, default=0,
                   help="Store, per step, the molecule's values for the N spectral features "
                        "most correlated with that PC, for the filmstrip's bar row. "
                        "0 omits the row.")
    p.add_argument("--n-background-scatter", type=int, default=50_000,
                   help="Points kept for each panel's grey backdrop. Percentiles, medians "
                        "and the nearest-molecule search always use every molecule; this "
                        "only thins the repeated visual backdrop. 0 keeps all.")
    p.add_argument("--corr-scatter-max", type=int, default=0,
                   help="Cap on points stored for the correlation scatters. 0 (default) "
                        "keeps every molecule, reproducing the old figures exactly, at "
                        "roughly 100 MB per 1M molecules. Reported r values always use "
                        "every molecule regardless.")
    return p.parse_args()


def main():
    args = parse_args()
    layerwise_dir = args.layerwise_dir or args.data_dir
    spectral_dir = args.spectral_dir or args.data_dir

    if args.list_embeddings:
        keys = available_decoder_keys(layerwise_dir, args.prefix, args.splits)
        print(f"global_cond")
        for stream, layer, timestep in keys:
            print(f"{'x' if stream == 'x_hidden_mean' else 'y'}:{layer}@{timestep:g}")
        print(f"\n{len(keys)} decoder embeddings + global_cond")
        return

    specs = parse_embedding_specs(args.embeddings, layerwise_dir, args.prefix, args.splits)
    encoder_specs = [s for s in specs if s.kind == "encoder"]
    decoder_specs = [s for s in specs if s.kind == "decoder"]

    print("=" * 78)
    print("Joint structural + spectral PCA")
    print(f"  data dir      : {args.data_dir}")
    print(f"  spectral dir  : {spectral_dir}")
    if decoder_specs:
        print(f"  layerwise dir : {layerwise_dir}")
    print(f"  prefix/splits : {args.prefix} / {args.splits}")
    print(f"  embeddings    : {len(specs)} ({len(encoder_specs)} encoder, "
          f"{len(decoder_specs)} decoder)")
    print(f"  n PCs         : {args.n_pcs}  (scaled: {not args.no_scale})")
    print("=" * 78)

    descriptor_names = list(args.descriptor_names or DEFAULT_DESCRIPTOR_NAMES)
    inputs = [Path(args.data_dir) / f"{args.prefix}_{s}_global_cond.pt" for s in args.splits]
    inputs += [Path(spectral_dir) / f"{args.prefix}_{s}_spectral_features.pt" for s in args.splits]
    if decoder_specs:
        inputs += [Path(layerwise_dir) / f"{args.prefix}_{s}_layerwise.pt" for s in args.splits]

    shared = dict(
        descriptor_names=descriptor_names,
        n_components=args.n_pcs,
        scale=not args.no_scale,
        seed=args.seed,
        top_k=args.top_k,
        n_steps=args.n_steps,
        pct_lo=args.pc_percentile_lo,
        pct_hi=args.pc_percentile_hi,
        n_background_scatter=args.n_background_scatter,
        n_bar_features=args.n_bar_features,
        corr_scatter_max=args.corr_scatter_max,
        slug=SLUG,
        inputs=inputs,
        params={"prefix": args.prefix, "splits": list(args.splits),
                "data_dir": str(args.data_dir)},
    )

    def tag_for(spec) -> str:
        return f"{spec.slug}_{args.tag_suffix}" if args.tag_suffix else spec.slug

    # --- encoder -------------------------------------------------------------
    if encoder_specs:
        print("\n### loading the encoder corpus ###")
        corpus = load_encoder_corpus(args.data_dir, spectral_dir, args.prefix, args.splits)
        print(f"Corpus: {len(corpus['smiles'])} molecules, embedding "
              f"{corpus['embedding'].shape}, spectral panel {corpus['spectral_features'].shape}")
        descriptors = cached_descriptor_matrix(
            corpus["smiles"], descriptor_names, args.n_desc_workers,
            use_cache=not args.no_descriptor_cache)

        for spec in encoder_specs:
            tag = tag_for(spec)
            print(f"\n### {spec.label} -> data/{SLUG}/{tag}/ ###")
            fit_joint_pca(
                embedding=corpus["embedding"], smiles=corpus["smiles"],
                dataset=corpus["dataset"], descriptor_matrix=descriptors,
                spectral_features=corpus["spectral_features"],
                spectral_feature_names=corpus["spectral_feature_names"],
                h_peaks=corpus["h_peaks"], c_peaks=corpus["c_peaks"], h_nh=corpus["h_nh"],
                peak_index=None,
                out_dir=data_dir(SLUG, tag, create=True), tag=tag,
                embedding_label=spec.label, **shared)

    # --- decoder -------------------------------------------------------------
    if decoder_specs:
        print("\n### loading the decoder corpus ###")
        corpus = load_decoder_corpus(args.data_dir, layerwise_dir, spectral_dir, args.prefix,
                                     args.splits, decoder_specs, n_workers=args.n_desc_workers)
        print(f"Corpus: {len(corpus['smiles'])} molecules matched to the cotrain corpus")
        descriptors = cached_descriptor_matrix(
            corpus["smiles"], descriptor_names, args.n_desc_workers,
            use_cache=not args.no_descriptor_cache)

        for spec in decoder_specs:
            tag = tag_for(spec)
            print(f"\n### {spec.label} -> data/{SLUG}/{tag}/ ###")
            fit_joint_pca(
                embedding=corpus["embeddings"][(spec.stream, spec.layer, spec.timestep)],
                smiles=corpus["smiles"], dataset=corpus["dataset"],
                descriptor_matrix=descriptors,
                spectral_features=corpus["spectral_features"],
                spectral_feature_names=corpus["spectral_feature_names"],
                h_peaks=corpus["h_peaks"], c_peaks=corpus["c_peaks"], h_nh=corpus["h_nh"],
                peak_index=corpus["peak_index"],
                out_dir=data_dir(SLUG, tag, create=True), tag=tag,
                embedding_label=spec.label, **shared)

    print(f"\nDone. Draw the figures with:\n"
          f"    python plotting/{SLUG}.py --tag {tag_for(specs[0])}")


if __name__ == "__main__":
    main()
