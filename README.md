# COMP3003 Individual Dissertation

Credits to the original code belong to Jonas Wacker @ https://github.com/joneswack/brats-pretraining published in the paper [Transfer Learning for Brain Tumor Segmentation](https://arxiv.org/abs/1912.12452) on arXiv.

This repository contains code to reproduce our proposed extension of the original project.

# Setup and Installation
To run this project, **Python 3.6 and above** must be installed on your system. We recommend downloading [Python 3.6.8](https://www.python.org/downloads/release/python-368/) from the official source as this is the official version of Python we use in our project. This project uses `PyTorch 1.5.1` with `CUDA 10.1`

## Recommended Hardware Specifications
Disk space: We **highly recommend** at least allocating ***40GB*** of disk space to store both preprocessed data and original data in the next few steps.

GPU: We use a **RTX2080Ti 11GB** for our experiments. A smaller GPU will affect the required batch size and a larger GPU will allow a larger batch size.

Memory: We recommend allocating 24GB of memory for this experiment.

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


# Dataset and Preprocessing
## Obtaining our dataset
We use the both the Training and Validation dataset from the Multimodal Brain Tumour Segmentation Challenge 2020 (BraTS2020) in our experiments. The Training dataset consists of **369** entries, while the Validation dataset consists of **125** entries. We are using the Validation dataset as our **Test** dataset due to the unavailability of the actual Test dataset.

The dataset folder can be downloaded [by clicking this link](www.google.com), and should be extracted into the root directory of this project. The resulting path for the dataset should be `brats-pretraining-jiachenn/dataset`

## Pre-processing our dataset
Our dataset is first preprocessed into a format that the code accepts by using the BraTS preprocessing example from [batchgenerators/examples/brats2017](https://github.com/MIC-DKFZ/batchgenerators/tree/master/batchgenerators/examples/brats2017) courtesy of Fabian Issense.

The first step is to check the `config.py` file. The file works with just the relative paths to the downloaded dataset and destination folder instead of requiring absolute paths.  However there are subtle differences when using a Windows-based system and a Unix-based system due to the foward-slash `/` used in Unix paths, and the backslash `\\` used in Windows paths.

For a Windows-based system, your `config.py` should look like this:

```
brats_preprocessed_destination_folder_train_2020 = "brats_data_preprocessed\\Brats20TrainingData"

brats_folder_with_downloaded_data_training_2020 = "dataset\\MICCAI_BraTS2020_TrainingData"

brats_preprocessed_destination_folder_test_2020 = "brats_data_preprocessed\\Brats20ValidationData"

brats_folder_with_downloaded_data_test_2020 = "dataset\\MICCAI_BraTS2020_ValidationData"

num_threads_for_brats_example = 8
```

For a Unix-based system, your `config.py` should look like this:
```
brats_preprocessed_destination_folder_train_2020 = "brats_data_preprocessed/Brats20TrainingData"

brats_folder_with_downloaded_data_training_2020 = "dataset/MICCAI_BraTS2020_TrainingData"

brats_preprocessed_destination_folder_test_2020 = "brats_data_preprocessed/Brats20ValidationData"

brats_folder_with_downloaded_data_test_2020 = "dataset/MICCAI_BraTS2020_ValidationData"

num_threads_for_brats_example = 8
```

Run the commands provided below to perform the preprocessing step. It is worth noting that the data is preprocessed into `.npy` files and takes up significantly more disk space (22GB preprocessed training data compared to 3GB for raw training data!!) compared to the original `.nii.gz` files. 

The preprocessing command is different for both Windows and Unix based systems due to their path differences.

For Unix-based systems (Linux, MacOS), run the following:
```
$ python preprocessing.py -type Training
$ python preprocessing.py -type Test 
```

For Windows-based systems, run the following:
```
$ python preprocessing.py -type Training -os Windows
$ python preprocessing.py -type Test -os Windows
```

All raw data will be preprocessed into the directory `brats_data_preprocessed/Brats20TrainingData` and `brats_data_preprocessed/Brats20ValidationData` respectively.


# Training and Predictions
## Running the training process 
We perform our training process on the University's GPU cluster that uses the Slurm scheduler. However, we provide commands for both running on the scheduler, or running it locally, provided you have a CUDA-enabled GPU.

Training is performed over 50 epochs and a model is saved after 50 epochs. The available arguments that are able to be passed to the train command can be displayed with `python main.py -h` which will bring up the following:

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

`--learning_rate`: Sets the learning rate for the model training (default: 1e-3)

`--epochs`: Number of training epochs (default: 50)

`--seed`: Can be a random value for PyTorch weight initialization

### Toggle-basis flags
These flags default to a certain value if not added into the training command e.g. putting `--no_gpu` in the training command uses CPU instead of GPU for training.

`--no_validation`: Choose whether to use validation set 

`--no_gpu`: If selected, uses CPU for training. Else, uses GPU

`--no_multiclass`: If enabled, only trains on Tumor Core, else trains on all classes provided.

`--no_pretrained`: Whether to use pretrained model or not

If you are running on the University's GPU cluster using the Slurm scheduler, use the command to run our configuration:

```
$ sbatch run_experiments_3d.sh
```

However if you are running locally with a CUDA-enabled device, use this command to run our configurations:
```
$ python3 main.py -name albunet_4_channels_1 --num_channels 4 --seed 1		
```

When training is completed, the model produced will be saved to `brats-pretraining-jiachenn/models` with the name specified in the `-name` argument.

## Producing segmentation output
To produce the segmentation output, we require a crucial piece of information, the **model name**. We have **standardized the model name** for ease of reproducing our results, hence the commands below can be ran as it is. Test set predictions are saved to `segmentation_output/<model name>`.

If you are running on the University's GPU cluster with the Slurm scheduler:
```
$ sbatch produce_seg_output_saved_models.sh
```

If you are running locally,

```
$ python3 run_saved_models.py -model_name albunet_4_channels_1  
```

# Evaluation of Results
To evaluate our results, the segmentation output has to be uploaded to [CBICA Image Processing Portal](https://ipp.cbica.upenn.edu). You must first create an account, and await approval from the administrators. Once the account has been approved, kindly select `BraTS'20 Validation Data: Segmentation Task` under the `MICCAI BraTS 2020` section, and upload the segmentation labels into the space provided. It will take some time for the portal to process the results, and the output will be a `.zip` file containing log files, and a `stats_validation_final.csv` file which contains our results.

Common issues: Sometimes the `.zip` file comes back without `stats_validation_final.csv`, in this case just create another job and upload the labels again.

A few Jupyter Notebooks have been created under `brats-pretraining_jiachenn/Jupyter_Notebooks` to provide interpretations of the results, the notebooks are as follows:

1. Dice-Plots.ipynb - Interprets our results
2. Read-Logs.ipynb - Provides graphs of training-validation training loss progress
3. Seg-Graphic.ipynb - Visualizes the segmentation output from `segmentation_output/` and provides options to save figures.

# Common Errors
1. CUDA out of memory error

    Try reducing the batch size from 12 to a lower number. We recommend using the RTX2080Ti 11GB for our processing.

# Expected Directory Structure TO BE UPDATED!

The structure of this repository should be as follows:

```
brats-pretraining_jiachenn
│   .gitignore
│   brats_data_loader.py
│   config.py
│   loss.py
│   main.py
│   models.py
│   preprocessing.py
│   produce_seg_output_saved_models.sh
│   README.md
│   run_experiments_3d.sh
│   run_saved_models.py
│   tb_log_reader.py
│   train_test_function.py
│
├───brats_data_preprocessed
│   ├───Brats20TrainingData
│   └───Brats20ValidationData
├───CSV_Results
├───dataset
│   ├───MICCAI_BraTS2020_TrainingData
│   └───MICCAI_BraTS2020_ValidationData
├───figures
├───Images
├───Jupyter_Notebooks
│       Dice-Plots.ipynb
│       Read-Logs.ipynb
│       Seg-Graphic.ipynb
│
├───models
├───segmentation_output
└───tensorboard_logs
```