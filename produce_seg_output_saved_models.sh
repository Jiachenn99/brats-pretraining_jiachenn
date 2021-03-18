#!/bin/sh

#SBATCH -p cs -A cs -q csug
#SBATCH -c8 --mem=24g
#SBATCH --gres gpu:1 --constraint=2080ti

module load nvidia/cuda-10.1
module load nvidia/cudnn-v7.6.5.32-forcuda10.1

for i in 20210316-184209_brats20_3d_pretrained_originalconfig_1_lr_0.001_epochs_50_epochbatch_100_epoch
do
	python3 run_saved_models.py -model_name $i --batch_size 12 --patch_depth 24 --brats_test_year 20
done
