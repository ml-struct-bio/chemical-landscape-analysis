## 1) Script overview

### 0_extr_ecfp_globalcond.py
What it does:
- Extracts SMILES strings, global-condition embeddings, ECFP fingerprints, and dataset/source labels for the cotrain dataset.
- Processes train/val/test splits and saves them in a single structured PyTorch file per split.

Inputs:
- path to the nmr-to-3d repo root
- checkpoint path
- checkpoint sigma value
- output directory and prefix
- optional source list / split overrides

Outputs:
- files like `cotrain_train_global_cond.pt`, `cotrain_val_global_cond.pt`, `cotrain_test_global_cond.pt`
- a JSON manifest with run metadata

---

### 0a_extr_decoder.py
What it does:
- Similar to the previous script, but also extracts decoder/trunk hidden states at selected layers and diffusion timesteps.

Inputs:
- same core inputs as above
- list of layers and timesteps to extract
- optional sampling cap per dataset

Outputs:
- files like `cotrain_train_layerwise.pt`, `cotrain_val_layerwise.pt`, `cotrain_test_layerwise.pt`
- a JSON manifest

---

### 1_cluster_dataset.py
What it does:
- Clusters molecules from the extracted embedding data using ECFP-based structure clustering (Taylor-Butina style).
- Produces cluster labels and cluster-level summaries.

Inputs:
- directory containing the extracted `.pt` files
- prefix and split(s)
- clustering parameters such as sample size and Butina cutoff

Outputs:
- `cotrain_cluster_labs.npy` (cluster labels aligned to the concatenated molecule order)
- `cotrain_cluster_meta.csv` (SMILES, dataset, split, cluster)
- `cluster_stats.csv` (cluster-level descriptor summaries)
- `cluster_representatives.png` (representative molecule grid, when generated)
- a clustering manifest JSON

---

### 1a_make_claude_prompt.py
What it does:
- Builds per-cluster prompt files for later LLM-based labeling or annotation.
- Uses cluster labels, representative molecules, and descriptor summaries.

Inputs:
- extracted embedding data
- cluster labels and cluster summary files
- minimum cluster size and number of representative/diverse examples

Outputs:
- `cluster_llm_prompts.csv` (one row per cluster, with prompt text)

---

### 2_trav_and_correlations.py
What it does:
- Fits PCA on the embedding space.
- Creates PC traversals by stepping through each component and showing nearby molecules.
- Computes correlations between PCs and RDKit descriptors.

Inputs:
- extracted embedding `.pt` files
- embedding key (default: `global_cond`)
- number of PCs and number of traversal steps

Outputs:
- `*_pc_descriptor_correlations.csv`
- `*_pc_descriptor_correlations.txt`
- `*_pc_best_descriptor.csv`
- `*_pca_correlations.png`
- `*_pc{1..N}_traversal.png`

---

### 3_embedding_semantics.py
What it does:
- Fits linear regression directions for molecular properties and creates property-direction traversals.
- Measures how well embedding similarity matches ECFP Tanimoto similarity via nearest-neighbor overlap.

Inputs:
- extracted embedding data and ECFP vectors
- property panel (for example MolWt, LogP, TPSA, RingCount, Aromaticity)

Outputs:
- `*_property_traversal_<property>.png`
- `*_property_regression_fits.csv`
- `*_nn_overlap_stats.txt`
- scatter plots like `*_embedding_vs_tanimoto_scatter.png`

---

### 4_pretty_plot.py
What it does:
- Creates a UMAP-style main figure for the embedding space.
- Can highlight clusters or specific SMILES molecules and make inset panels for those regions.

Inputs:
- extracted embedding data
- optional cluster label file
- optional precomputed 2D UMAP embedding
- optional list of SMILES to highlight
- optional dataset subset filter

Outputs:
- `umap_main.png` (or a custom main figure name)
- `region_cluster_<id>.png` and `region_smiles_<label>.png`
- `region_*_mols.png` for sampled molecules in the highlighted regions
- `umap_figure_manifest.json`
- optional cached UMAP embedding `.npy`

---

### 5_property_plotss.py
What it does:
- Fits a UMAP on the train split and projects val/test into the same space.
- Visualizes train/val/test together and separately.
- Quantifies distribution shift with centroid shifts, nearest-neighbor distance, and KS tests.
- Plots embeddings colored by molecular properties.

Inputs:
- train/val/test embedding data
- optional dataset filter
- optional precomputed UMAP reducer

Outputs:
- `*_splits_combined.png`
- `*_split_train_by_dataset.png`, `*_split_val_by_dataset.png`, `*_split_test_by_dataset.png`
- `*_distribution_shift.txt`
- `*_property_<property>.png`
- optional pickled UMAP reducer

---

### 6_real_v_synthetic.py
What it does:
- Compares real-spectrum vs. synthetic-spectrum embeddings for paired datasets.
- Measures paired vs. background embedding distance, retrieval accuracy, and property correlation.

Inputs:
- nmr-to-3d repo root
- checkpoint and sigma value
- real/synthetic dataset pair definitions
- chosen split (usually test)

Outputs:
- per-pair plots such as:
  - `*_pca_drift.png`
  - `*_distance_hist.png`
  - `*_property_correlation.png`
- overall plots:
  - `retrieval_accuracy_by_pair.png`
- text report: `real_vs_synthetic_gap_report.txt`

---

### 7_general_dataset_stats.py
What it does:
- Loads the extracted cotrain `.pt` files for the requested splits.
- Splits molecules by source dataset and computes a compact set of RDKit molecular descriptors.
- Aggregates general statistics per dataset and saves both tabular summaries and publication-style plots.

Inputs:
- directory containing `<prefix>_<split>_global_cond.pt` files
- optional dataset subset filter
- output directory and number of worker processes

Outputs:
- `<prefix>_dataset_stats.csv`
- `<prefix>_dataset_counts.png`
- `<prefix>_descriptor_boxplots.png`
- `<prefix>_descriptor_histograms.png`

---

## 2) Possible output files and figures this folder can generate

The exact filenames depend on the `--prefix`, `--out-dir`, and other arguments, but the following are the main artifact types this folder can produce.

### Embedding / dataset artifacts
- `cotrain_train_global_cond.pt`
- `cotrain_val_global_cond.pt`
- `cotrain_test_global_cond.pt`
- `cotrain_train_layerwise.pt`
- `cotrain_val_layerwise.pt`
- `cotrain_test_layerwise.pt`
- `*_manifest.json`

### Clustering outputs
- `cluster_summary/cotrain_cluster_labs.npy`
- `cluster_summary/cotrain_cluster_meta.csv`
- `cluster_summary/cluster_stats.csv`
- `cluster_summary/cluster_llm_prompts.csv`
- `cluster_summary/cluster_representatives.png`

### PCA / traversal outputs
- `pca_summary/*_pc_descriptor_correlations.csv`
- `pca_summary/*_pc_descriptor_correlations.txt`
- `pca_summary/*_pc_best_descriptor.csv`
- `pca_summary/*_pca_correlations.png`
- `pca_summary/*_pc{1..N}_traversal.png`

### Property / semantic analysis outputs
- `property_and_nn_summary/*_property_traversal_<property>.png`
- `property_and_nn_summary/*_property_regression_fits.csv`
- `property_and_nn_summary/*_nn_overlap_stats.txt`
- `property_and_nn_summary/*_embedding_vs_tanimoto_scatter.png`

### UMAP / visualization outputs
- `umap_figure/umap_main.png`
- `umap_figure/region_cluster_*.png`
- `umap_figure/region_smiles_*.png`
- `umap_figure/region_*_mols.png`
- `umap_figure/umap_figure_manifest.json`
- optional cached UMAP `.npy` files

### Split-shift / property-map outputs
- `umap_splits_summary/*_splits_combined.png`
- `umap_splits_summary/*_split_*_by_dataset.png`
- `umap_splits_summary/*_distribution_shift.txt`
- `umap_splits_summary/*_property_<property>.png`
- optional pickled UMAP reducer files

### Real vs. synthetic comparison outputs
- `real_vs_synthetic_summary/*_pca_drift.png`
- `real_vs_synthetic_summary/*_distance_hist.png`
- `real_vs_synthetic_summary/*_property_correlation.png`
- `real_vs_synthetic_summary/retrieval_accuracy_by_pair.png`
- `real_vs_synthetic_summary/real_vs_synthetic_gap_report.txt`

### General dataset statistics outputs
- `dataset_stats/*_dataset_stats.csv`
- `dataset_stats/*_dataset_counts.png`
- `dataset_stats/*_descriptor_boxplots.png`
- `dataset_stats/*_descriptor_histograms.png`

## 3) Typical workflow
A typical analysis sequence is:

1. run `0_extr_ecfp_globalcond.py` to create base embedding files
2. optionally run `0a_extr_decoder.py` for layerwise features
3. run `1_cluster_dataset.py` to cluster molecules
4. run `1a_make_claude_prompt.py` if LLM-based annotation is needed
5. run `2_trav_and_correlations.py`, `3_embedding_semantics.py`, `4_pretty_plot.py`, or `5_property_plotss.py` for analysis and figures
6. run `6_real_v_synthetic.py` for real-vs-synthetic comparisons
7. run `7_general_dataset_stats.py` for high-level molecular dataset summaries and plots
