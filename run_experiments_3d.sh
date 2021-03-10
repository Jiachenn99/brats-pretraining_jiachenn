#!/bin/sh

#SBATCH -p cs -A cs -q csug
#SBATCH -c12 --mem=24g
#SBATCH --gres gpu:2 --constraint=2080ti

module load nvidia/cuda-10.1
module load nvidia/cudnn-v7.6.5.32-forcuda10.1

for i in 1
do
	for j in 1e-1 1e-4 1e-5
	do
		python3 train_jonas_net_batch.py -name brats20_3d_pretrained_new_lr_$j_$i  --batch_size 12 --patch_depth 24 --brats_train_year 20 --brats_test_year 20 --epochs 60 --learning_rate $j --seed $i
	done

# Brats18 data
#python3 train_jonas_net_batch.py -name brats18_3d_pretrained_1  --batch_size 10 --patch_depth 24 --brats_train_year 18 --brats_test_year 18 --epochs 20 --multi_gpu --seed 1

# Brats20 data
#python3 train_jonas_net_batch.py -name brats20_3d_pretrained_new_$i  --batch_size 12 --patch_depth 24 --brats_train_year 20 --brats_test_year 20 --epochs 60 --learning_rate 1e-2 --seed $i
#python3 train_jonas_net_batch.py -name brats20_3d_pretrained_new_$i  --batch_size 12 --patch_depth 24 --brats_train_year 20 --brats_test_year 20 --epochs 2 --seed $i

done
