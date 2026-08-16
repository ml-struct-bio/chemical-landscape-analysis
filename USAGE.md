# USAGE.md — quick command guide

Full docs/rationale are in [README.md](README.md); this is just "what do I type."

## 0. Setup

```bash
source /usr/licensed/anaconda3/2025.6/etc/profile.d/conda.sh
conda activate nmr3d
cd /scratch/gpfs/ZHONGE/jc4587/research/1_chemical_landscape_analysis/8_14_26_cotrainv3_dedupOFF_s0_epoch1399

DATA=/scratch/gpfs/ZHONGE/jc4587/research/cotrainv3_embeddings/26-07-27-cotrain-v3-dedupOFF-s0/epoch1399
```

Extraction (`extraction/00`/`01`/`02`) is already done for `train`/`val`/`test` in `$DATA`. Everything
below just reads those files.

**Golden rule: use `--splits train` everywhere.** Cluster labels, PCA, UMAP, and the layer comparison
all cross-reference each other *positionally* by which `--splits` they were built over — mixing splits
between steps means a downstream step either refuses (row-count mismatch) or silently mis-aligns. The
full-corpus UMAP fits already cached (see below) and `04_joint_pca`'s existing 72-tag decoder sweep were
both built on `train`, so staying on `train` also means you hit those caches instead of recomputing.

## 1. The non-UMAP analyses

Each is `analysis/NN_x.py` (compute) → `plotting/NN_x.py` (draw). Analysis writes `data/NN_x/<tag>/`;
plotting reads it and writes `figures/NN_x/<tag>/`. Re-running plotting alone is always free (no
recomputation) — only re-run the analysis half if you want different numbers.

```bash
# 03 — Butina clustering (do this first; everything else can colour by it)
python analysis/03_clustering.py --data-dir "$DATA" --splits train --tag main
python plotting/03_clustering.py --tag main

# 04 — joint structural+spectral PCA (global_cond shown; --embeddings all-decoder
#       for every decoder layer — already done for all 72 in this repo, see data/04_joint_pca/)
python analysis/04_joint_pca.py --data-dir "$DATA" --splits train --embeddings global_cond
python plotting/04_joint_pca.py --all-tags   # redraws every tag already computed

# 06 — per-dataset molecular characterization
python analysis/06_dataset_stats.py --data-dir "$DATA" --splits train
python plotting/06_dataset_stats.py --tag main

# 07 — decoder layers vs. ECFP/global_cond baselines (pick one stream per run)
python analysis/07_layer_comparison.py --data-dir "$DATA" --layerwise-dir "$DATA" \
    --splits train --stream x_hidden_mean --cluster-tag main --traversal-layers 1 6 11
python plotting/07_layer_comparison.py --tag main
# for the other stream: --stream y_hidden_mean --tag y_stream, then plot --tag y_stream
```

`03_clustering/val` already in this repo was built on `--splits val`, not `train` — it won't align with
a `train`-split UMAP/layer-comparison run. Re-run it with `--tag main --splits train` as above if you
want cluster colouring alongside everything else.

## 2. UMAP — where the flexibility lives

Same two-step contract, but the analysis stores the 2-D projection **plus every column a figure might
colour by** in one pass, so plotting can draw one figure or a hundred without ever refitting.

### 2a. Which embedding — `--embeddings` (analysis side)

| value | meaning |
|---|---|
| `global_cond` | the peak embedder's pooled output (encoder) |
| `ecfp` | Morgan fingerprint — a model-free structural baseline |
| `x:5@0.001` | decoder atom/coord stream, layer 5, timestep 0.001 |
| `y:11@1.0` | decoder NMR stream, layer 11, timestep 1.0 |
| `x:5` | ...first timestep available for that layer |
| `all-decoder` | every (stream, layer, timestep) in the layerwise file — 72 on this corpus |
| `all` | `global_cond` + `ecfp` + `all-decoder` |

`--list-embeddings` prints what's actually available. Each embedding gets its own `--tag`
(`data/05_umap/global_cond/`, `.../decoder_x_L05_t0.001/`, ...).

### 2b. Which colouring — `--color-by` (plotting side)

| value | meaning |
|---|---|
| `none` | flat grey (or `--point-color`) |
| `dataset` / `sim_real` / `split` / `cluster` | categorical (`cluster` needs `--cluster-tag` at analysis time) |
| any property name, e.g. `MolWt` | whatever `--properties` computed at analysis time |
| any spectral feature, e.g. `n_C_peaks` | from `extraction/01`'s panel |
| `all-properties` | one figure per property |
| `all-spectral` | one figure per spectral feature |
| `all` | literally everything above, in one command |

`--list-colorings --tag <tag>` prints what a given tag supports. `--all-tags` draws every tag under
`data/05_umap/` in one command.

### 2c. Every embedding × every colouring — the actual recipe

Because of `--embeddings all` / `--color-by all` / `--all-tags`, the full matrix for the two baselines is
**two commands** (the `global_cond`/`ecfp` UMAP fits are already cached from the previous pipeline's
`umap_shared_cache/` at default hyperparameters — this hits that cache, it does not refit):

```bash
# 1. compute + store every colour column, for both baselines, over the FULL train corpus
python analysis/05_umap.py --data-dir "$DATA" --splits train \
    --embeddings global_cond ecfp --properties extended --cluster-tag main

# 2. draw literally everything for both
python plotting/05_umap.py --all-tags --color-by all
```

That's ~1 (none) + 4 (categorical) + 34 (`extended` properties) + 37 (spectral features) ≈ **76 PDFs per
embedding tag**, drawn in seconds once step 1 has run (step 2 never refits).

### 2d. Extending to decoder layers

`--embeddings all-decoder` (or `all`) adds up to 72 more tags. Unlike the baselines, there is **no
pre-existing cache** for decoder UMAPs — each is a fresh fit. It's cheaper than it sounds, though: decoder
embeddings are the ~75,000-molecule extraction subsample, not the full 2.5M corpus, so each fit is minutes,
not hours. Still, 72 back-to-back adds up. Pick one:

```bash
# a specific handful you actually care about
python analysis/05_umap.py --data-dir "$DATA" --splits train \
    --embeddings x:1 x:6 x:11 y:1 y:6 y:11 --properties extended --cluster-tag main

# or everything, as a batch job (same shape as the existing 04_joint_pca all-decoder sweep)
python analysis/05_umap.py --data-dir "$DATA" --splits train \
    --embeddings all-decoder --properties extended --cluster-tag main
```

Then the same one-liner draws everything: `python plotting/05_umap.py --all-tags --color-by all`.

### 2e. Extras

At analysis time (needs a re-run of `analysis/05_umap.py` to add):

```bash
--highlight-smiles "aspirin=CC(=O)OC1=CC=CC=C1C(=O)O" --knn 15   # mark molecules + cosine nearest neighbours
--inset-clusters 0 3988 31        # or --inset-top-clusters 3     # zoomed panels for Butina cluster regions
```

At plot time (free, no re-run):

```bash
--insets                                   # draw the inset figure (needs the above)
--highlight 0 3988 31                      # colour only these labels of ONE categorical colouring, grey the rest
--dataset-colors nmrexp='#4C8DAE'          # override configs/colors.yaml for one run
--cluster-colors / --sim-real-colors / --anchor-colors   # same idea, other palette modes
--max-points 200000                        # thin what's DRAWN only; stored projection is untouched
```

## Tests

```bash
python -m pytest tests/
```
