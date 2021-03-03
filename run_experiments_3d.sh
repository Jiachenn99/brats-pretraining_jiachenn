#!/bin/sh

#SBATCH -p cs -A cs -q csug
#SBATCH -c12 --mem=24g
#SBATCH --gres gpu:2 --constraint=2080ti

module load nvidia/cuda-10.1
module load nvidia/cudnn-v7.6.5.32-forcuda10.1

for i in 1 2 3 4 5
do

echo "Current i(seed) in script:" $i

# Multi gpu
python3 train_jonas_net_batch.py -name brats20_3d_pre_$i --batch_size 10 --patch_depth 24 --brats_train_year 20 --brats_test_year 20 --epochs 50 --multi_gpu --seed $i
python3 train_jonas_net_batch.py -name brats20_3d_no_pretrained_$i --batch_size 10 --patch_depth 24 --brats_train_year 20 --brats_test_year 20 --epochs 50 --multi_gpu --no_pretrained --seed $i

# Multi gpu testing
#python3 train_jonas_net_batch.py -name brats20_3d_pre_$i --batch_size 10 --patch_depth 24 --brats_train_year 20 --brats_test_year 20 --epochs 10 --multi_gpu --seed $i

# No multi gpu testing
#python3 train_jonas_net_batch.py -name brats20_3d_pre_$i --batch_size 10 --patch_depth 24 --brats_train_year 20 --brats_test_year 20 --epochs 2 --seed $i brats20_3d_pre_1_lr_0.001_epochs_50
done
