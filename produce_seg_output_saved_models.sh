#!/bin/sh

#SBATCH -p cs -A cs -q csug
#SBATCH -c12 --mem=24g
#SBATCH --gres gpu:2 --constraint=2080ti

module load nvidia/cuda-10.1
module load nvidia/cudnn-v7.6.5.32-forcuda10.1

#for i in 20210309-091413_brats20_3d_pretrained_new_lr_1_lr_0.1_epochs_60 20210309-133958_brats20_3d_pretrained_new_lr_1_lr_0.0001_epochs_60 20210309-170344_brats20_3d_pretrained_new_lr_1_lr_1e-05_epochs_60
#for i in 20210309-100704_brats20_3d_pretrained_new_lr_1_lr_0.1_epochs_60 20210309-021115_brats20_3d_pretrained_new_1_lr_0.01_epochs_60
for i in 20210309-133958_brats20_3d_pretrained_new_lr_1_lr_0.0001_epochs_60_50
do
	python3 run_saved_models.py -model_name $i --batch_size 12 --patch_depth 24 --brats_test_year 20
done
