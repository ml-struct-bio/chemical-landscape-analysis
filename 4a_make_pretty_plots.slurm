#!/bin/bash
#SBATCH --output=/home/jc4587/3_AI4chemistr/FINAL/4a_make_pretty_plots/out.out
#SBATCH --error=/home/jc4587/3_AI4chemistr/FINAL/4a_make_pretty_plots/err.err
#SBATCH --job-name=pretty_plots

#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=01:30:00

#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=jc4587@princeton.edu

module purge
module load anaconda3/2025.6
conda activate nmr3d

python 4_pretty_plot.py \
    --data-dir /scratch/gpfs/ZHONGE/jc4587/nmr_embs_FINAL \
    --color-by np_class \
    --np-class-labels-path /scratch/gpfs/ZHONGE/jc4587/nmr_embs_FINAL/labels.pkl \
    --np-class-probs-path /scratch/gpfs/ZHONGE/jc4587/nmr_embs_FINAL/labels_probs.pkl \
    --np-class-min-probability 0.99 \
    --np-class-top-n 10 \
    --smiles-of-interest \
        "CC1=C2[C@H](C(=O)[C@@]3([C@H](C[C@@H]4[C@]([C@H]3[C@@H]([C@@](C2(C)C)(C[C@@H]1OC(=O)[C@@H]([C@H](C5=CC=CC=C5)NC(=O)C6=CC=CC=C6)O)O)OC(=O)C7=CC=CC=C7)(CO4)OC(=O)C)O)C)OC(=O)C" \
        "CC(C)C[C@H](C(=O)N[C@@H]1[C@@H](C2=CC(=C(C=C2)OC3=CC4=CC(=C3O)OC5=C(C=C(C=C5)[C@H]([C@H]6C(=O)N[C@@H](C7=CC(=CC(=C7C8=C(C=CC(=C8)[C@H](C(=O)N6)NC(=O)[C@@H]4NC(=O)[C@@H](NC1=O)CC(=O)N)O)O)O)C(=O)O)O)Cl)Cl)O)NC" \
        "NC1=N[C@H](O)[C@H]2[C@H]3O[C@]4(O)O[C@@H](C(O)[C@@]2(N1)[C@@H]4O)[C@]3(O)CO" \
        "C[C@@H]1CC[C@H]2C[C@@H](/C(=C/C=C/C=C/[C@H](C[C@H](C(=O)[C@@H]([C@@H](/C(=C/[C@H](C(=O)C[C@H](OC(=O)[C@@H]3CCCCN3C(=O)C(=O)[C@@]1(O2)O)[C@H](C)C[C@@H]4CC[C@H]([C@@H](C4)OC)O)C)/C)O)OC)C)C)/C)OC" \
        "CN1CC[C@]23[C@@H]4[C@H]1CC5=C2C(=C(C=C5)O)O[C@H]3[C@H](C=C4)O" \
        "C[C@@H]1CC[C@H]2[C@H](C(=O)O[C@H]3[C@@]24[C@H]1CC[C@](O3)(OO4)C)C" \
        "CCN(CC)C(=O)[C@H]1CN([C@@H]2CC3=CNC4=CC=CC(=C34)C2=C1)C" \
        "CN1CC(C=C2C1CC3=CNC4=CC=CC2=C34)C(=O)O" \
    --out-dir 4a_make_pretty_plots \
    --main-fig-name np_classes_showcase
