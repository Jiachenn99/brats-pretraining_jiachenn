#!/bin/sh

#SBATCH -p cs -A cs -q csug
#SBATCH -c8 --mem=32g
#SBATCH --gres gpu:1 --constraint=2080ti

module load nvidia/cuda-10.1
module load nvidia/cudnn-v7.6.5.32-forcuda10.1

#20210321-014705_brats20_3d_small_training_1_lr_0.0001_epochs_70_epochbatch_75_epoch
for i in 20210321-070858_brats20_3d_small_training_2_lr_0.0001_epochs_70_epochbatch_75_epoch 
do
	python3 run_saved_models.py -model_name $i --batch_size 12 --patch_depth 24 --brats_test_year 20 --epochs_max 60 --testing_train_set 1

done


#for i in 20210321-095418_brats20_3d_pretrained_trainbatchsize_100_2_lr_0.0001_epochs_50_epochbatch_100_epoch 
#do
#	python3 run_saved_models.py -model_name $i --batch_size 12 --patch_depth 24 --brats_test_year 20
#done



#for j in 20 30 40 
#do
#	for i in 20210309-133958_brats20_3d_pretrained_new_lr_1_lr_0.0001_epochs_60_$j
#	do
#		python3 augmented_test_data.py -model_name $i --batch_size 12 --patch_depth 24 --brats_test_year 20 --epochs_max 20 --patient_start 0 --patient_end 25
#		python3 augmented_test_data.py -model_name $i --batch_size 12 --patch_depth 24 --brats_test_year 20 --epochs_max 20 --patient_start 26 --patient_end 51
#		python3 augmented_test_data.py -model_name $i --batch_size 12 --patch_depth 24 --brats_test_year 20 --epochs_max 20 --patient_start 52 --patient_end 77
#		python3 augmented_test_data.py -model_name $i --batch_size 12 --patch_depth 24 --brats_test_year 20 --epochs_max 20 --patient_start 77 --patient_end 102
#		python3 augmented_test_data.py -model_name $i --batch_size 12 --patch_depth 24 --brats_test_year 20 --epochs_max 20 --patient_start 103 --patient_end 125
#	done
#done
