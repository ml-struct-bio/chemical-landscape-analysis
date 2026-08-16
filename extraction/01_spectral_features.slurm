#!/bin/bash

#SBATCH --output=/scratch/gpfs/ZHONGE/jc4587/research/1_chemical_landscape_analysis/8_14_26_cotrainv3_dedupOFF_s0_epoch1399/extraction/01_spectral_features.out
#SBATCH --error=/scratch/gpfs/ZHONGE/jc4587/research/1_chemical_landscape_analysis/8_14_26_cotrainv3_dedupOFF_s0_epoch1399/extraction/01_spectral_features.err
#SBATCH --job-name=spectral_feats

#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=4:00:00

#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=jc4587@princeton.edu

# No --gres=gpu on purpose: spectral features come off the datamodule's peak
# dict, so the model is never constructed. The RDKit structure-reference pass
# is what wants the 16 cores.
#
# --mem is sized for the train split: reading step 0's file to get its molecule
# order pulls the whole thing in, embeddings included (~13 GB on disk, more
# once unpickled), on top of the extracted panel. Run val/test alone and 64G is
# plenty.

module purge
module load anaconda3/2025.6
conda activate nmr3d

cd /scratch/gpfs/ZHONGE/jc4587/research/1_chemical_landscape_analysis/8_14_26_cotrainv3_dedupOFF_s0_epoch1399

python extraction/01_spectral_features.py \
    --nmr3d-root /scratch/gpfs/ZHONGE/jc4587/research/1_chemical_landscape_analysis/8_14_26_cotrainv3_dedupOFF_s0_epoch1399/nmr-to-3d \
    --corpus-dir /scratch/gpfs/ZHONGE/jc4587/research/cotrainv3_embeddings/26-07-27-cotrain-v3-dedupOFF-s0/epoch1399 \
    --prefix cotrain \
    --splits train val test \
    --n-workers 14
