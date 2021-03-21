#!/bin/sh

#SBATCH -p cs -A cs -q csug
#SBATCH -c8 --mem=24g
#SBATCH --gres gpu:1 --constraint=2080ti

module load nvidia/cuda-10.1
module load nvidia/cudnn-v7.6.5.32-forcuda10.1

for i in 1 2 3
do
	python3 train_jonas_net_batch.py -name brats20_3d_small_training_$i --batch_size 12 --patch_depth 24 --brats_train_year 20 --brats_test_year 20 --learning_rate 0.0001 --training_batch_size 75 --validation_batch_size 75 --training_max 300 --epochs 70 --seed $i		
	python3 train_jonas_net_batch.py -name brats20_3d_pretrained_trainbatchsize_100_$i --batch_size 12 --patch_depth 24 --brats_train_year 20 --brats_test_year 20 --learning_rate 0.0001 --epochs 50 --seed $i		

done

# Brats18 data
#python3 train_jonas_net_batch.py -name brats18_3d_pretrained_1  --batch_size 10 --patch_depth 24 --brats_train_year 18 --brats_test_year 18 --epochs 20 --multi_gpu --seed 1

# Brats20 data
#python3 train_jonas_net_batch.py -name brats20_3d_pretrained_new_$i  --batch_size 12 --patch_depth 24 --brats_train_year 20 --brats_test_year 20 --epochs 2 --seed $i

