# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repository.

## What this is

A ground-up rebuild of `../8_11_26_cotrainv3_dedupOFF_s0_epoch1399/`, an
analysis pipeline for embeddings from an NMR-to-molecule prediction transformer.
The old repo is the source material and reference implementation; it is **not**
on the import path and must not be imported from.

The rebuild exists for one reason: to make computation and figure generation
separate programs. Analyses are being ported one at a time, on request, and some
are being merged or dropped along the way. **Do not bulk-port.** Right now only
`extraction/` exists.

See [README.md](README.md) for the layout and the full contract.

## The rule that matters

Analysis scripts write `data/<slug>/<tag>/`; plotting scripts read it and write
`figures/<slug>/<tag>/`. They are separate processes and share nothing else.

When porting an analysis, the work is mostly deciding **what the figure actually
needs** and moving everything else to the analysis side. The old pipeline stored
raw N×D embeddings and pickled sklearn/UMAP estimators in its artifacts so plots
could run nearest-neighbour searches and projections at draw time. Here, if the
plot code would have to search, fit, or project, the analysis does it and stores
the answer. Figure data is `.npz` / `.csv` / `.json` only — no pickles, no `.pt`,
no raw embeddings.

There is no `--plot-only` flag anywhere. Replotting is running the plot script.

Consequences worth remembering:
- Analysis scripts take no cosmetic arguments; plot scripts take no `--data-dir`.
- `src/analysis/` must never import matplotlib; `src/plotting/` must never
  import sklearn/umap/torch or anything under `src.analysis`. Shared colour
  logic lives in `src/common/palette.py`, shared molecule rendering in
  `src/plotting/mol_render.py`.
- `cache/` (pickled refits, shared UMAP fits) is analysis-only.
- `tests/test_layer_boundaries.py` enforces all of the above by AST scan. If a
  port needs to violate it, that is a signal the split is wrong, not the test.

## Shared GPU

The GPU on this machine is shared with other users' jobs — a run can find only
a fraction of its 40 GB free. Any analysis that puts a large intermediate on the
GPU should size its chunks from the actual tensor shape (not a molecule count),
catch `torch.cuda.OutOfMemoryError`, halve, and fall back to CPU rather than
losing the run. `assign_to_prototypes` in `src/analysis/clustering.py` is the
worked example: its memory scales with chunk × n_prototypes, so the caller's
`--assign-chunk-size` is treated as a cap, not the operative value.

## Conventions to preserve

- Embedding filenames: `<prefix>_<split>_global_cond.pt`,
  `<prefix>_<split>_spectral_features.pt`, `<prefix>_<split>_layerwise.pt`,
  with `split` ∈ `{train, val, test}`.
- Traversal filmstrips step across the **1st–99th percentile** of an axis, not
  true min/max, and always mark each step with a **real molecule's own
  coordinates** — never an interpolated, non-existent point.
- `src/common/ckpt_meta.py` is the single source of truth for
  checkpoint-derived data settings. Never reintroduce hardcoded per-milestone
  `sigma_data`. Note the asymmetry it encodes: `dataset_args.sigma_data` does
  not reach the model at eval time, so a wrong value corrupts only the manifest
  — whereas `c_peak_norm_args` *is* applied at val/test and does change the
  extracted tensors.
- The hydra config caps test splits at **100 molecules** with a null seed
  (`dataset_args.test_args.test_samples`). Any extraction of a `test` split must
  override it or be silently truncated. `01_spectral_features.py` does this;
  copy that handling into anything new that embeds a test split.
  A consequence worth knowing: because `00_global_cond` and `02_decoder_layers`
  each drew their own null-seeded 100 per source, their **`test` splits are
  different samples** — real overlap on this corpus is 1 molecule out of 300.
  Anything joining those two files must use `train`/`val`.
  `src/analysis/corpus.py` refuses a low-match join rather than analyzing the
  accidental overlap.
- Scripts run from `analysis/`, `plotting/` and `extraction/`, i.e. from a
  subdirectory, so each prepends the repo root to `sys.path` before its
  `from src....` imports. That prelude must stay above those imports.
- `src/` has **no `__init__.py`** and must not gain one. It is a PEP 420
  namespace package on purpose: `src/extraction/nmr3d.py`'s
  `nmr3d_import_scope` swaps `src` over to `--nmr3d-root`'s own *regular* `src`
  package by evicting the cached module, and an `__init__.py` here breaks that
  with a `ModuleNotFoundError` that points nowhere near the cause.

## Environment

```bash
source /usr/licensed/anaconda3/2025.6/etc/profile.d/conda.sh
conda activate nmr3d
```

HPC/SLURM. `.slurm` files sit beside the script they submit. Full-`train`
extraction and full-corpus UMAP fits belong in batch jobs, not on a shared login
node; the `test` split is fine interactively.

## Tests

```bash
python -m pytest tests/
```

Plain `tests/test_*.py`, no pytest config. `tests/conftest.py` puts the repo
root on `sys.path`.
