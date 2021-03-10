#!/bin/sh

#SBATCH -p cs -A cs -q csug
#SBATCH -c12 --mem=24g
#SBATCH --gres gpu:2 --constraint=2080ti

module load nvidia/cuda-10.1
module load nvidia/cudnn-v7.6.5.32-forcuda10.1

for i in 1
do
	python3 train_jonas_net_batch.py -name brats20_3d_pretrained_FORGOTTENLR_$i_lr_0.001 --batch_size 12 --patch_depth 24 --brats_train_year 20 --brats_test_year 20 --epochs 50 --seed $i 
	for j in 50 100 200
	do
		python3 train_jonas_net_batch.py -name brats20_3d_pretrained_trainbatchsize_$j --batch_size 12 --patch_depth 24 --brats_train_year 20 --brats_test_year 20 --epochs 50 --seed $i --training_batch_size $j --validation_batch_size $j
	done
done

# Brats18 data
#python3 train_jonas_net_batch.py -name brats18_3d_pretrained_1  --batch_size 10 --patch_depth 24 --brats_train_year 18 --brats_test_year 18 --epochs 20 --multi_gpu --seed 1

# Brats20 data
#python3 train_jonas_net_batch.py -name brats20_3d_pretrained_new_$i  --batch_size 12 --patch_depth 24 --brats_train_year 20 --brats_test_year 20 --epochs 2 --seed $i

