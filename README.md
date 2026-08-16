# Cotrain NMR embedding analysis — rebuild

Analysis pipeline for embeddings from an NMR-to-molecule prediction transformer
(cotrain-v3, `dedupOFF-s0`, epoch 1399).

This is a ground-up rebuild of `../8_11_26_cotrainv3_dedupOFF_s0_epoch1399/`
with one structural change: **computation and figures are fully separate
programs.** Analyses are being ported one at a time, on request.

For a condensed command reference ("what do I actually type"), see
[USAGE.md](USAGE.md); this file covers the layout, the contract, and each
analysis's rationale in more depth.

## Layout

```
extraction/   GPU + checkpoint. Produces the embedding/feature tensors.
analysis/     Compute entrypoints.  NN_slug.py  ->  data/NN_slug/<tag>/
plotting/     Figure entrypoints.   NN_slug.py  <-  data/NN_slug/<tag>/
                                                ->  figures/NN_slug/<tag>/
src/
  extraction/ model + datamodule + spectral-feature libraries
  analysis/   computation libraries    (never import matplotlib)
  plotting/   figure libraries         (never import sklearn/umap/torch)
  common/     paths, manifests, palette, checkpoint metadata
data/         figure data      (gitignored)
figures/      PDFs             (gitignored)
cache/        refit objects    (gitignored, analysis-only)
```

An analysis and its plots share a number and slug — `analysis/04_pca.py` pairs
with `plotting/04_pca.py` — and communicate only through
`data/04_pca/<tag>/`. `--tag` (default `main`) separates parameter variants;
re-running either script overwrites its own tag in place.

## The contract

**Analysis scripts** take `--data-dir/--prefix/--splits/--tag` plus compute
arguments, and write `data/<slug>/<tag>/`. They take no cosmetic arguments and
there is no `--plot-only` flag — replotting is just running the plotting script.

**Plot scripts** take `--tag` plus cosmetic arguments only. They load
`data/<slug>/<tag>/`, and write PDFs to `figures/<slug>/<tag>/`. They never read
`--data-dir`, never open `cache/`, and never re-fit anything.

**Figure data is small and plain.** `.npz` for arrays, `.csv` for tables,
`.json` for metadata — no pickles, no `.pt`, no raw N×D embedding matrices. If a
figure would need a search, a fit, or a projection, the *analysis* does it and
stores the answer. Concretely: a UMAP analysis stores 2-D coordinates and label
codes (~10 MB at N=1M), not the 512-dimensional embeddings and a pickled fit
(GBs); a PCA traversal stores the SMILES it already resolved for each step, not
the matrices needed to resolve them again.

Every data directory carries a `manifest.json` — schema version, input files,
parameters, git SHA — written *last*, so its presence means the run finished.
Plot scripts check it and tell you what to re-run if it is missing or stale.

`cache/` is the one place pickles live: shared UMAP fits and similar refit
objects, so an analysis can resume without re-reading multi-GB tensors. Only
`analysis/` scripts may touch it.

## Environment

```bash
source /usr/licensed/anaconda3/2025.6/etc/profile.d/conda.sh
conda activate nmr3d
```

HPC (SLURM). `.slurm` files sit next to the script they submit. Full-`train`
extraction and full-corpus UMAP fits are batch jobs, not login-node work; the
small `test` split is safe interactively.

`nmr-to-3d/` is the model-training repo, needed by everything in `extraction/`
as `--nmr3d-root`. It is a symlink to a single checkout shared across every
dated analysis directory
(`/scratch/gpfs/ZHONGE/jc4587/research/1_chemical_landscape_analysis/nmr-to-3d`),
not a copy local to this one; replace the symlink with your own clone if you
want this repo's checkout to diverge from the shared one.

## Extraction

Run these first — every analysis consumes their output. All three write to
`--save-dir`, which is the `--data-dir` everything downstream reads.

| Script | Output | Needs |
|---|---|---|
| `extraction/00_global_cond.py` | `<prefix>_<split>_global_cond.pt`, `<prefix>_manifest.json` | GPU + ckpt |
| `extraction/01_spectral_features.py` | `<prefix>_<split>_spectral_features.pt` | CPU only |
| `extraction/02_decoder_layers.py` | `<prefix>_<split>_layerwise.pt`, `<prefix>_layerwise_manifest.json` | GPU + ckpt |

`00` extracts SMILES, peak-embedder `global_cond` embeddings, ECFP fingerprints
and source-dataset labels. `01` adds the per-molecule spectral feature panel
plus the raw ¹H/¹³C peak lists, aligned row-for-row with `00`'s output — it reads
`00`'s manifest for its configuration rather than restating it, and matches
molecules by `(dataset, SMILES)` so alignment is checked, not assumed. `02`
extracts decoder trunk hidden states at chosen layers and diffusion timesteps.

These three are carried over from the previous pipeline **unchanged apart from
imports and docstrings** — they are deliberately not refactored into the
analysis/plotting split, and they produce byte-compatible outputs.

```bash
sbatch extraction/00_global_cond.slurm     # then 01, then 02
```

### Checkpoint-derived settings

`src/common/ckpt_meta.py` is the single source of truth. `sigma_data` and the
¹³C `c_peak_norm_args` normalization are read from the checkpoint's own
`hyper_parameters`, never hardcoded per milestone. The asymmetry matters:
`dataset_args.sigma_data` does **not** reach the model at eval time (the
checkpoint restores `diffusion_process_args.sigma_data` itself), so a wrong
value corrupts only the manifest — whereas `c_peak_norm_args` *is* applied at
val/test and does change the extracted tensors. `--ckpt-sigma` survives only as
an assertion.

## Analyses

### `03_clustering` — Butina clusters over ECFP

Butina is exact and quadratic, so it runs on a **stratified subsample**
(proportional per source dataset), keeps each cluster's Butina centroid as a
prototype fingerprint, and assigns the whole corpus to its nearest prototype by
Tanimoto. The subsampled molecules keep their exact Butina label rather than a
nearest-prototype approximation of it.

```bash
python analysis/03_clustering.py --data-dir "$DATA" --prefix cotrain --splits train \
    --n-cluster-sample 10000 --butina-cutoff 0.35
python plotting/03_clustering.py --tag main
```

Outputs in `data/03_clustering/<tag>/`:

| file | what |
|---|---|
| `cluster_labels.npy` | int32, one label per molecule, in extraction row order |
| `descriptors.npz` | MolWt / LogP / TPSA / Rings per molecule |
| `cluster_stats.csv` | per cluster: size + mean descriptors, largest first |
| `plot_clusters.csv` | the top-N clusters the figure draws, bars pre-normalized |
| `representatives.csv` | sampled SMILES per plotted cluster |

Figures: `cluster_representatives.pdf` (structures + descriptor bars per
cluster) and `cluster_sizes.pdf` (cluster-size distribution plus the
distribution of each mean descriptor across ALL clusters, not just the
largest few the representatives figure draws).

`cluster_labels.npy` is the cross-analysis product here — later steps colour by
cluster and compare cluster quality across decoder layers.

**The Butina step is cached** under `cache/butina/`, keyed on corpus size,
source composition, subsample size, cutoff, seed and ECFP settings. Re-running
with different plotting or descriptor options skips the expensive part;
`--refit` forces it. This replaces the old hand-managed `--load-prototypes`
pickle, which is still accepted for reproducing an older clustering.

`--n-cluster-sample` is the one knob that can quietly ruin a run: Butina needs
the entire lower triangle in memory, ~32 bytes per pair, so 10k molecules costs
~1.6 GB and 100k would need ~160 GB. The analysis refuses sizes past that ceiling
up front rather than dying hours in.

The old flat `cluster_meta.csv` is opt-in via `--write-meta-csv`. It is a
~500 MB file at full corpus scale whose only new columns are `cluster` and the
four descriptors — both already saved compactly above, aligned to extraction
row order.

### `04_joint_pca` — joint structural + spectral PCA

Fits ONE PCA over a chosen embedding and explains its components in two
vocabularies at once: RDKit structural descriptors, and NMR spectral features.
The previous pipeline fit that PCA twice (its scripts `3` and `17`), so its two
"PC3" figures were only the same axis by coincidence of a shared seed.

The embedding is selectable — encoder, any decoder trunk hidden state, or all of
them in one run. Each gets its own tag.

```bash
DATA=/scratch/gpfs/ZHONGE/jc4587/research/cotrainv3_embeddings/26-07-27-cotrain-v3-dedupOFF-s0/epoch1399

python analysis/04_joint_pca.py --data-dir "$DATA" --splits train \
    --embeddings global_cond x:5@0.001 y:11@1.0
python plotting/04_joint_pca.py --all-tags
```

| `--embeddings` | meaning |
|---|---|
| `global_cond` | the peak embedder's pooled output (encoder) |
| `x:5@0.001` | decoder atom/coord stream, layer 5, timestep 0.001 |
| `y:11@1.0` | decoder NMR stream, layer 11, timestep 1.0 |
| `x:5` | ... first timestep available for that layer |
| `all-decoder` | every (stream, layer, timestep) — 72 on this corpus |
| `all` | `global_cond` + `all-decoder` |

`--list-embeddings` prints what the layerwise file actually contains. Tags are
`global_cond`, `decoder_x_L05_t0.001`, `decoder_y_L11_t1`, …; `--tag-suffix`
keeps parameter variants side by side.

Figures: `pc{i}_joint_traversal.pdf` (¹H sticks / ¹³C sticks / structure /
[feature bars] / scatter, every panel the same real molecule),
`descriptor_correlations.pdf`, `spectral_correlations.pdf`, and a correlation
heatmap for each vocabulary.

**Decoder embeddings are joined to the corpus.** `extraction/02` subsamples, so
its molecules are a subset of the `global_cond` corpus in a different order.
SMILES, dataset labels, spectral features and peaks are all indexed through a
join that prefers `mol_idx`, falls back to raw SMILES, then canonical SMILES for
whatever is left. If under half the rows match, the run **fails** rather than
analyzing a small mismatched subset.

> Decoder embeddings do not work on `--splits test`. Both extractions drew their
> own random 100 molecules per source (the null-seed test cap), so the two test
> splits are simply different samples — actual overlap on this corpus is 1
> molecule out of 300. Use `train` or `val`; the error message explains this.

Scale, for a 75,000-molecule decoder run: **2.8 MB** of figure data per
embedding. The equivalent old artifact carried the raw 75000×768 embedding, the
full descriptor and spectral matrices, and pickled `PCA`/`StandardScaler`
objects.

### `05_umap` — general-purpose UMAP maps

Projects any embedding to 2-D and stores **every column a figure might colour
by** beside it, so colourings are pure plot-time choices. Replaces the previous
pipeline's `5`, `6` and `7`, which re-ran the whole load-and-fit once per
colouring via a batch driver that shelled out to itself.

```bash
python analysis/05_umap.py --data-dir "$DATA" --splits train \
    --embeddings global_cond ecfp --properties extended
python plotting/05_umap.py --tag global_cond --color-by dataset cluster MolWt
```

**Embeddings** — same grammar as `04_joint_pca`, plus `ecfp`:
`global_cond`, `ecfp`, `x:5@0.001`, `y:11@1.0`, `x:5`, `all-decoder`, `all`.
Each gets its own tag. `--list-embeddings` shows what's available.

**Colourings** (`--color-by`, repeatable):

| | |
|---|---|
| `none` | all one colour (`--point-color`, grey by default) |
| `dataset` | source dataset of the cotrain mixture |
| `sim_real` | real vs. synthetic vs. unknown (`--real-prefixes`/`--sim-prefixes`) |
| `split` | train / val / test |
| `cluster` | Butina cluster id from `03_clustering` |
| any property | e.g. `MolWt`, `TPSA` — `--properties basic\|extended\|all` |
| any spectral feature | e.g. `n_C_peaks`, `C_shift_mean` |
| `all-properties`, `all-spectral`, `all` | one figure each |

`--list-colorings` prints what a given tag supports.

`--highlight` colours only the named labels and greys the rest — how a few
clusters stay legible against millions of points. It applies to whichever
categorical colouring actually contains those labels, so mixing
`--color-by dataset cluster MolWt --highlight 0 3988 31` works; naming labels
that match nothing is an error.

**Colours** resolve through `configs/colors.yaml` plus `--dataset-colors`,
`--cluster-colors`, `--sim-real-colors`, so a label keeps its colour across
every figure in the pipeline. Palette entries naming a label absent from a
particular figure are ignored with a note rather than failing — Butina
renumbers clusters on each re-run, so pinned ids go stale.

Marker size and alpha default to values scaled by point count; `--point-size`
and `--alpha` override. `--max-points` thins what is drawn without touching the
stored projection.

#### Reusing the previous pipeline's fits

UMAP over the full corpus costs hours, so fits are cached by a hyperparameter
fingerprint. `--reuse-cache-dir` defaults to
`../8_11_26_cotrainv3_dedupOFF_s0_epoch1399/umap_shared_cache/`, which already
holds **full-corpus (2,524,941 molecule) fits for `global_cond` (cosine, scaled)
and `ecfp` (jaccard, unscaled)** at the default hyperparameters. The
fingerprinting is ported verbatim, so those are found and loaded rather than
refit; new fits are written to this repo's `cache/umap/` and never to the
read-only one. `--no-reuse-cache` ignores it.

The fingerprint is a **config** hash — hyperparameters plus point count and
dimensionality — not a content hash. A different corpus with the same shape and
settings will match a stale entry, so use a separate `--umap-cache-dir` per
corpus if that is a risk.

#### Highlighting molecules, nearest neighbours, and insets

```bash
python analysis/05_umap.py --data-dir "$DATA" --splits train --embeddings global_cond \
    --highlight-smiles "aspirin=CC(=O)OC1=CC=CC=C1C(=O)O" --knn 15 \
    --inset-top-clusters 3 --n-region-mols 6

python plotting/05_umap.py --tag global_cond --color-by dataset --insets \
    --anchor-colors aspirin=crimson
```

`--highlight-smiles` takes `name=SMILES` or a bare SMILES (`--highlight-smiles-file`
for a longer list, one per line, `#` comments allowed); matching is by
**canonical SMILES**, so the string need not be written the same way as the
corpus. `--knn N` finds each anchor's `N` nearest neighbours by **cosine
similarity in the high-dimensional embedding** — not in the 2-D map, which
would beg the question by returning a tight visual blob every time regardless
of how the real space looks. Where those neighbours land on the map is exactly
the point: dispersed neighbours mean UMAP distorted that neighbourhood.

`--inset-clusters ID [ID ...]` or `--inset-top-clusters N` (the N largest) add
Butina cluster regions alongside any anchors — both draw from `03_clustering`'s
labels via `--cluster-tag`.

Anchors, their neighbours, and each inset region's representative molecules are
all analysis output (`anchors.csv`, `anchor_neighbors.csv`, `regions.csv`,
`region_mols.csv`) — bounded in size regardless of corpus, since only a handful
of anchors and a capped sample per region are kept.

`plotting/05_umap.py --insets` draws each region as a zoomed scatter beside a
structure grid, with the region boxed on the main map; omit it to just mark
anchors without the inset panels (`umap_anchors.pdf`). Region colours resolve
through the `anchor` palette mode, same mechanism as everything else.

### `06_dataset_stats` — per-dataset molecular characterization

Characterizes and compares the corpus's source datasets: the 34-descriptor
RDKit panel plus element composition, ring topology, stereochemistry, Murcko
scaffold diversity, and exact cross-dataset molecule overlap. Port of the
previous pipeline's `9_run_dataset_stats_experiment.py`, molecular half only —
its spectral half needs a checkpoint/GPU and script 17's extraction machinery,
which has not been ported here yet.

```bash
python analysis/06_dataset_stats.py --data-dir "$DATA" --prefix cotrain --splits train \
    --max-per-dataset 200000
python plotting/06_dataset_stats.py --tag main
```

Two passes, split by cost: an exact pass over the **full corpus** on canonical
SMILES alone (dataset sizes, cross-dataset overlap), and the expensive
34-descriptor + composition/ring/stereo/scaffold pass on a **stratified
subsample** (`--max-per-dataset`, default 200k/dataset — every output here is a
distribution, so beyond that a sample and the full corpus are indistinguishable).

Outputs in `data/06_dataset_stats/<tag>/`: `stats.csv` (the sampled per-molecule
table every distribution figure reads), `counts.csv`, `summary.csv`,
`divergence.csv` (per-feature KS + standardized mean difference, every dataset
pair), `overlap_pairs.csv` / `overlap_summary.csv` (exact, full corpus),
`scaffold_summary.csv` / `scaffold_top.csv` / `scaffold_coverage.csv`.

Figures: dataset sizes, descriptor boxplots/histograms/violins/ECDFs and their
correlation matrices, a divergence heatmap (features ranked by how strongly
they separate any dataset pair), element composition, ring topology,
stereochemistry, scaffold diversity, cross-dataset overlap, and a MolWt-vs-LogP
chemical-space hexbin per dataset.

Dataset colors resolve through `configs/colors.yaml`'s `dataset` block (same
mechanism as `05_umap`'s `--color-by dataset`), so a source keeps the same
color here as everywhere else in the pipeline; override per-run with
`--dataset-colors nmrexp='#4C8DAE'`.

### `07_layer_comparison` — decoder layers vs. ECFP/global_cond baselines

Compares decoder per-layer hidden-state representations against each other and
against two baselines (ECFP, the peak embedder's `global_cond`): property
linear-decodability, unsupervised cluster-quality vs. Butina labels,
ECFP-vs-embedding nearest-neighbor agreement, and PC1/PC2-vs-RDKit-descriptor
interpretability -- all swept across decoder depth -- plus PC-traversal
filmstrips for a handful of representative layers.

```bash
python analysis/07_layer_comparison.py --data-dir "$DATA" --layerwise-dir "$DATA" \
    --prefix cotrain --splits train --stream x_hidden_mean \
    --cluster-tag main --traversal-layers 1 6 11
python plotting/07_layer_comparison.py --tag main
```

Port of the previous pipeline's `10_run_layer_comparison_experiment.py`, with
two things deliberately left out: property-direction traversal (this port only
does PC traversal, reusing the same `geometry.traversal_steps` machinery as
`04_joint_pca`), and the old per-representation PC1/PC2-vs-descriptor scatter
grid (~38 near-duplicate figures on a full sweep) -- the PC-interpretability
SWEEP chart plus the full `pc_correlations.csv` numbers cover the same ground
without one PDF per layer.

`--stream` picks ONE decoder stream per run (`x_hidden_mean` or
`y_hidden_mean`, matching the previous pipeline's CLI); compare the other
stream with a second run under a different `--tag`. `--cluster-tag` points at
a `data/03_clustering/<tag>/cluster_labels.npy` run over the SAME `--splits`
(cluster-quality metrics are skipped if omitted) -- gathered through the same
join the decoder embeddings came through, same mechanism `05_umap`'s decoder
branch uses.

Outputs in `data/07_layer_comparison/<tag>/`: `metrics.csv` (one row per
representation), `summary.txt` (best representation per metric),
`pc_correlations.csv` / `pc_best_descriptor.csv`, `traversal.csv` /
`traversal_stats.csv` (PC-traversal filmstrip steps, real molecules only), and
one subsampled `traversal_background_<representation>.npz` per selected
traversal layer.

**Loading and aligning every (layer, timestep) of one stream is cached** under
`cache/layer_comparison/`, keyed on `(layerwise_dir, data_dir, prefix, splits,
stream)` -- it is by far the most expensive step here (tens of GB read plus a
corpus-wide join via `src/analysis/corpus.py`'s `align_to_corpus`) and has
nothing to do with which metrics or traversal layers are requested, so
re-running with different `--traversal-layers`/`--n-steps`/percentiles reuses
it. `--refit` forces a redo. As with the UMAP cache, the fingerprint is a
CONFIG hash, not a content hash -- swapping in different data under the same
directories/prefix/splits/stream will false-positive match a stale entry.

## Tests

```bash
python -m pytest tests/
```

`tests/test_layer_boundaries.py` enforces the import rules above by AST-scanning
the source tree. It is the reason the separation is expected to hold this time:
in the previous pipeline the same split existed only as a convention, and seven
plotting modules had drifted into importing from `src/analysis/`.
