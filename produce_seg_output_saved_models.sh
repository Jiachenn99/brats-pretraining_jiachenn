#!/bin/sh

#SBATCH -p cs -A cs -q csug
#SBATCH -c8 --mem=32g
#SBATCH --gres gpu:1 --constraint=2080ti

module load nvidia/cuda-10.1
module load nvidia/cudnn-v7.6.5.32-forcuda10.1

for i in albunet_4_channels_1_epoch_50
do
	python3 run_saved_models.py -model_name $i --brats_test_year 20 --num_channels 4
done


