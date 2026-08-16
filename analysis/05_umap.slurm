#!/bin/bash

#SBATCH --output=/scratch/gpfs/ZHONGE/jc4587/research/1_chemical_landscape_analysis/8_14_26_cotrainv3_dedupOFF_s0_epoch1399/analysis/05_umap.out
#SBATCH --error=/scratch/gpfs/ZHONGE/jc4587/research/1_chemical_landscape_analysis/8_14_26_cotrainv3_dedupOFF_s0_epoch1399/analysis/05_umap.err
#SBATCH --job-name=umap_decoder_x

#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00

#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=jc4587@princeton.edu

# Every decoder ATOM/COORD-stream (x) embedding: 12 layers x 3 timesteps = 36
# UMAP fits, one output tag each. The NMR-conditioning stream (y) is skipped.
#
# No --gres=gpu: this reads cached extraction output and fits UMAP on CPU.
#
# Sizing notes:
#   --mem=64G   `cotrain_train_layerwise.pt` holds BOTH streams for all 36
#               pairs (75,000 x 768 float32 each, ~17 GB) and torch.load pulls
#               the whole file in before load_decoder_corpus drops the y half.
#               Peak is that plus the ~8 GB of x-stream kept, plus UMAP's own
#               working set.
#   --cpus-per-task=16  feeds UMAP's neighbour search and the RDKit property
#               pass (--n-property-workers defaults to a safe count).
#   --time=12:00:00  the decoder corpus is SUBSAMPLED by
#               extraction/02_decoder_layers.py (--n-samples-per-dataset 25000,
#               so 75,000 molecules for train, not the full 2.5M), which is what
#               makes 36 sequential fits affordable at all. Budgeted long
#               because the fits run one after another.
#
# Restarting is cheap. Each fit is written to cache/umap/ keyed by a
# fingerprint of its hyperparameters, and each tag's data/05_umap/<tag>/ is
# written as it completes -- so a job that dies at fit 20 re-runs and replays
# the first 19 from cache in seconds rather than refitting them.
#
# --real-prefixes/--sim-prefixes are NOT the defaults ('real'/'syn', which suit
# the real-*/syn-* eval benchmark pairs). On the cotrain training mixture only
# nmrexp is genuine experimental NMR; spectranp and uspto are both predicted.
# Without these the sim_real colouring classifies all three as 'unknown' and
# draws a uniformly grey figure.

module purge
module load anaconda3/2025.6
conda activate nmr3d

cd /scratch/gpfs/ZHONGE/jc4587/research/1_chemical_landscape_analysis/8_14_26_cotrainv3_dedupOFF_s0_epoch1399

DATA=/scratch/gpfs/ZHONGE/jc4587/research/cotrainv3_embeddings/26-07-27-cotrain-v3-dedupOFF-s0/epoch1399

# Built in a loop rather than typed out: 36 tokens by hand is how a missing
# space turns into a parse error 20 minutes into a queued job.
EMBS=()
for L in $(seq 0 11); do
    for T in 0.001 0.5 1; do
        EMBS+=("x:${L}@${T}")
    done
done

echo "Embeddings (${#EMBS[@]}): ${EMBS[*]}"

python analysis/05_umap.py \
    --data-dir "$DATA" \
    --splits train \
    --embeddings "${EMBS[@]}" \
    --cluster-tag main \
    --real-prefixes nmrexp \
    --sim-prefixes spectranp uspto
