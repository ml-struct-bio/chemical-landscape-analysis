#!/bin/bash

#SBATCH --output=/home/jc4587/3_AI4chemistr/FINAL/0a_extr_decoder.out
#SBATCH --error=/home/jc4587/3_AI4chemistr/FINAL/0a_extr_decoder.err
#SBATCH --job-name=extr_embs

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=jc4587@princeton.edu

module purge
module load anaconda3/2025.6
conda activate nmr3d

python 0a_extr_decoder.py --nmr3d-root /home/jc4587/3_AI4chemistr/nmr-to-3d --ckpt /projects/CRYOEM/zhonglab/data_nmr/2026/ckpts/26-05-01-cotraining-baselines/cotrain-epoch0899-accuracy60_70.ckpt --ckpt-sigma 2.8268 --save-dir /scratch/gpfs/ZHONGE/jc4587/nmr_embs_FINAL/ --layers -1 0 1 2 3 4 5 6 7 8 9 10 11 --timesteps 0.001 --n-samples-per-dataset 0 --splits train val test 