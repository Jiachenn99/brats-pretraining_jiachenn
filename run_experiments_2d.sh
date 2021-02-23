#!/bin/sh

#SBATCH -p cs -A cs -q csug
#SBATCH -c4 --mem=16000m

# for i in 1 2 3 4 5
# do
# python3 train_jonas_net_batch.py -name brats17_2d_pre_$i --batch_size 64 --patch_depth 1 --brats_train_year 17 --seed $i
python3 train_jonas_net_batch.py -name brats20_2d_pre_$i --batch_size 64 --patch_depth 1 --brats_train_year 20 --seed $i

# python3 train_jonas_net_batch.py -name brats17_2d_$i --batch_size 64 --patch_depth 1 --brats_train_year 17 --no_pretrained --seed $i
# done


