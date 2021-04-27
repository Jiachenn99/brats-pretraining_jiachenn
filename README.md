# COMP3003 Individual Dissertation

Credits to the original code belong to Jonas Wacker @ https://github.com/joneswack/brats-pretraining with the paper [Transfer Learning for Brain Tumor Segmentation](https://arxiv.org/abs/1912.12452) on arXiv.

This repository contains code to reproduce our proposed extension of the original project.

# Setup and Installation
To run this project, Python 3.6 and above **must** be installed on your system. We recommend downloading [Python 3.6.8](https://www.python.org/downloads/release/python-368/) from the official source as this is the official version of Python we use in our project. This project uses `PyTorch 1.5.1` with `CUDA 10.1`

Note: We **highly recommend** at least allocating ***40GB*** of disk space to store both preprocessed data and original data.

## Setting up PyTorch and CUDA

To install the version of PyTorch and CUDA that our project uses, simply run the following commands in sequence:

```
$ module load nvidia/cuda-10.1
$ module load nvidia/cudnn-v7.6.5.32-forcuda10.1
$ python -m pip install --upgrade pip
$ SRC=https://download.pytorch.org/whl/torch_stable.html
$ pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f $SRC
```

This will install `PyTorch 1.5.1` and `CUDA 10.1` from the source specified.

## Install all other dependencies
To set up all the dependencies required to run this project, the following commands can be ran in your terminal (Command Prompt, Bash) in the **root directory** of this project `brats-pretraining_jiachenn`. 

If you are using `pip` as your package manager, run the following command to install our dependencies.

```
$ python -m pip install --upgrade pip
$ pip install -r requirements.txt
```

If you are using a package manager such as `Anaconda/Conda`, we can create a virtual environment and install all files into the environment with the following command:
```
$ conda create --name virtualenv --file requirements.txt
```
>> Include instructions for windows and mac OS / linux

# Dataset and Preprocessing
## Obtaining our dataset
We use the both the Training and Validation dataset from the Multimodal Brain Tumour Segmentation Challenge 2020 (BraTS2020) in our experiments. The Training dataset consists of **369** entries, while the Validation dataset consists of **125** entries. We are using the Validation dataset as our **Test** dataset due to the unavailability of the actual Test dataset.

The dataset folder can be downloaded [by clicking this link](www.google.com), and should be extracted into the root directory of this project. The resulting path for the dataset should be `brats-pretraining-jiachenn/dataset`

## Pre-processing our dataset
Our dataset is first preprocessed into a format that the code accepts by using the BraTS preprocessing example from [batchgenerators/examples/brats2017](https://github.com/MIC-DKFZ/batchgenerators/tree/master/batchgenerators/examples/brats2017) courtesy of Fabian Issense.

The command provided below can be used to run to perform the preprocessing process. It is worth noting that the data is preprocessed into `.npy` files and takes up significantly more disk space (22GB preprocessed training data compared to 3GB for raw training data!!) compared to the original `.nii.gz` files. 
```
$ insert preprocessing command here and make options for both windows and linux/maOS systems
```
All raw data will be preprocessed into the directory `brats_data_preprocessed/Brats20TrainingData` and `brats_data_preprocessed/Brats20ValidationData` respectively.


# Training and Predictions
## Running the training process 
We perform our training process on the University's GPU cluster that uses the Slurm scheduler. However, we provide commands for both running on the scheduler, or running it locally, provided you have a CUDA-enabled GPU.

Training is performed over 50 epochs and a model is saved after 50 epochs. The arguments that are able to be passed to the train command are divided into two sections as follows:

### Require value as arguments flags

The general format for these flags are `-flag <value> or --flag <value>, e.g. --epochs 50`

`-name `: Name of the model

`--batch_size`: Batch size (default: 12 for RTX2080Ti 11GB)

`--patch_depth`: Patch depth to reisze image to (default: 24)

`--patch_width`: Patch width to reisze image to (default: 128)

`--patch_height`: Patch height to reisze image to (default: 128)

<!-- `--training_max`: Maximum number of  -->

`--training_batch_size`: Size of training minibatch (default: 100)

`--validation_batch_size`: Size of validation minibatch (default: 100)

<!-- `--brats_train_year`: 

`--brats_test_year`: -->

`--learning_rate`: Sets the learning rate for the model training

`--epochs`: Number of training epochs

`--seed`: Can be a random value for PyTorch weight initialization

### Toggle-basis flags
These flags default to a certain value if not added into the training command e.g. putting `--no_gpu` in the training command uses CPU instead of GPU for training.

`--no_validation`: Choose whether to use validation set 

`--no_gpu`: If selected, uses CPU for training. Else, uses GPU

`--no_multiclass`: If enabled, only trains on Tumor Core, else trains on all classes provided.

`--no_pretrained`: Whether to use pretrained model or not

If you are running on the University's GPU cluster using the Slurm scheduler, use the command:

```
$ sbatch run_experiments_3d.sh
```

However if you are running locally with a CUDA-enabled device, use the command to train with GPU enabled and all default settings:
```
$ python3 main.py -name <model name> --epochs 50 --seed 1		
```



When training is completed, the model produced will be saved to `brats-pretraining-jiachenn/models` with the name specified in the `-name` argument.

## Producing segmentation output
To produce the segmentation output, we require a crucial piece of information, the **model name**. The command below executes the prediction process and saves test data predictions to `segmentation_output/<model name>`

```
$ python3 run_saved_models.py -model_name $i 
```

# Expected Directory Structure TO BE UPDATED!

The structure of this repository should be as follows:


```
brats-pretraining_jiachenn
│   .gitignore
│   augmented_test_data.py
│   brats_data_loader.py
│   cj_net.py
│   config.py
│   Dice-Plots.ipynb
│   jonas_net.py
│   loss.py
│   main.py
│   preprocessing.py
│   produce_seg_output_saved_models.sh
│   Read-Logs.ipynb
│   README.md
│   run_experiments_3d.sh
│   run_saved_models.py
│   Seg-Graphic.ipynb
│   tb_log_reader.py
│   ternaus_unet_models.py
│   train_test_function.py
│   Trying_Resize.ipynb
│
├───brats_data_preprocessed
│   ├───Brats20TrainingData
│   └───Brats20ValidationData
├───figures
├───models
├───segmentation_output
└───tensorboard_logs
