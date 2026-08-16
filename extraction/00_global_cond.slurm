#!/bin/bash

#SBATCH --output=/scratch/gpfs/ZHONGE/jc4587/research/1_chemical_landscape_analysis/8_14_26_cotrainv3_dedupOFF_s0_epoch1399/extraction/00_global_cond.out
#SBATCH --error=/scratch/gpfs/ZHONGE/jc4587/research/1_chemical_landscape_analysis/8_14_26_cotrainv3_dedupOFF_s0_epoch1399/extraction/00_global_cond.err
#SBATCH --job-name=extr_globalcond

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00

#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=jc4587@princeton.edu

# Step 0 of the pipeline -- run this FIRST, everything else consumes its output.
#
# Uncapped by design (no --limit): all three v3 sources x train/val/test, i.e.
# ~2.5M molecules for train alone. Sizing notes:
#   --mem=128G      the merged train global_cond tensor alone lands around 13GB,
#                   and it is accumulated in memory before being written.
#   --cpus-per-task=8  feeds --n-ecfp-workers (RDKit Morgan fingerprints on CPU,
#                   the non-GPU half of the work).
#   --time=12:00:00 one peak-embedder forward pass per batch; far cheaper per
#                   batch than step 1's trunk passes, but over the full corpus.
#
# sigma_data and the 13C c_peak_norm_args normalization are NOT passed here --
# both are read from the checkpoint's own hyper_parameters (see
# src/common/ckpt_meta.py). The job prints what it resolved before extracting;
# check that line against the intended checkpoint.
#
# Smoke-test first. A capped run exercises every hydra suffix without paying
# for the corpus, and prints per-source train/val/test counts you can check
# against the v3 yaml's stated numbers:
#   python extraction/00_global_cond.py ... --limit 100 --save-dir /tmp/extr_smoke

module purge
module load anaconda3/2025.6
conda activate nmr3d

cd /scratch/gpfs/ZHONGE/jc4587/research/1_chemical_landscape_analysis/8_14_26_cotrainv3_dedupOFF_s0_epoch1399

python extraction/00_global_cond.py \
    --nmr3d-root /scratch/gpfs/ZHONGE/jc4587/research/1_chemical_landscape_analysis/8_14_26_cotrainv3_dedupOFF_s0_epoch1399/nmr-to-3d \
    --ckpt /projects/CRYOEM/zhonglab/data_nmr/2026/ckpts/26-07-27-cotrain-v3-dedupOFF-s0/epoch1399-accuracy75.98.ckpt \
    --save-dir /scratch/gpfs/ZHONGE/jc4587/research/cotrainv3_embeddings/26-07-27-cotrain-v3-dedupOFF-s0/epoch1399 \
    --prefix cotrain \
    --splits train val test \
    --n-ecfp-workers 8
