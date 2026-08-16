#!/bin/bash

#SBATCH --output=/scratch/gpfs/ZHONGE/jc4587/research/1_chemical_landscape_analysis/8_14_26_cotrainv3_dedupOFF_s0_epoch1399/extraction/02_decoder_layers.out
#SBATCH --error=/scratch/gpfs/ZHONGE/jc4587/research/1_chemical_landscape_analysis/8_14_26_cotrainv3_dedupOFF_s0_epoch1399/extraction/02_decoder_layers.err
#SBATCH --job-name=extr_embs

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00

#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=jc4587@princeton.edu

module purge
module load anaconda3/2025.6
conda activate nmr3d

cd /scratch/gpfs/ZHONGE/jc4587/research/1_chemical_landscape_analysis/8_14_26_cotrainv3_dedupOFF_s0_epoch1399

python extraction/02_decoder_layers.py \
    --nmr3d-root /scratch/gpfs/ZHONGE/jc4587/research/1_chemical_landscape_analysis/8_14_26_cotrainv3_dedupOFF_s0_epoch1399/nmr-to-3d \
    --ckpt /projects/CRYOEM/zhonglab/data_nmr/2026/ckpts/26-07-27-cotrain-v3-dedupOFF-s0/epoch1399-accuracy75.98.ckpt \
    --save-dir /scratch/gpfs/ZHONGE/jc4587/research/cotrainv3_embeddings/26-07-27-cotrain-v3-dedupOFF-s0/epoch1399 \
    --layers 0 1 2 3 4 5 6 7 8 9 10 11 --timesteps 0.001 0.5 1 --aug-conf conf10 --splits train val test --n-samples-per-dataset 25000
