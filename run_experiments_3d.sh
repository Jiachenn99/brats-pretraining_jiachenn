#!/bin/sh

#SBATCH -p cs -A cs -q csug
#SBATCH -c8 --mem=24g
#SBATCH --gres gpu:1 --constraint=2080ti

module load nvidia/cuda-10.1
module load nvidia/cudnn-v7.6.5.32-forcuda10.1

for i in 1
do
	python3 main.py -name albunet_4_channels_$i --batch_size 12 --training_max 20 --epochs 50 --num_channels 4 --seed $i		
done
