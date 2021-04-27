
#%%
from brats_data_loader import get_list_of_patients, get_train_transform, iterate_through_patients, BRATSDataLoader, get_train_transform_aggro
from train_test_function import ModelTrainer
from models import AlbuNet3D34, AlbuNet3D34_4channels
from loss import GeneralizedDiceLoss, SimpleDiceLoss, dice_multi_class, dice

from batchgenerators.utilities.data_splitting import get_split_deterministic
from batchgenerators.dataloading import MultiThreadedAugmenter


#%%
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

#%%
# videos.py
import argparse
parser = argparse.ArgumentParser(description='Train AlbuNet3D')
parser.add_argument('-name', type=str, help='Name of the Model')
parser.add_argument('--batch_size', type=int, help='Batch Size', default=24)
parser.add_argument('--patch_depth', type=int, help='Depth of the Input Patch', default=24)
parser.add_argument('--patch_width', type=int, help='Width of the Input Patch', default=128)
parser.add_argument('--patch_height', type=int, help='Height of the Input Patch', default=128)
parser.add_argument('--training_max', type=int, help='max number of patients for training', default=369)
parser.add_argument('--training_batch_size', type=int, help='Size of minibatch in training', default=100)
parser.add_argument('--validation_batch_size', type=int, help='Size of minibatch in validation', default=100)
parser.add_argument('--num_channels', type=int, help='Number of input channels', default=3)
parser.add_argument('--no_pretrained', dest='pretrained', action='store_false', help='ResNet34 without Pretraining')
parser.set_defaults(pretrained=True)
parser.add_argument('--brats_train_year', type=int, help='BRATS Train Year', default=17)
parser.add_argument('--brats_test_year', type=int, help='BRATS Test Year', default=18)
parser.add_argument('--no_validation', dest='use_validation', action='store_false', help='No Validation Set')
parser.set_defaults(use_validation=True)
parser.add_argument('--learning_rate', type=float, help='Learning Rate', default=1e-3)
parser.add_argument('--epochs', type=int, help='Number of Training Epochs', default=50)
parser.add_argument('--no_gpu', dest='use_gpu', action='store_false', help='Use CPU instead of GPU')
parser.set_defaults(use_gpu=True)
parser.add_argument('--no_multiclass', dest='multi_class', action='store_false', help='Tumor Core Only')
parser.set_defaults(multi_class=True)
parser.add_argument('--seed', type=int, help='PyTorch Seed for Weight Initialization', default=1234)
args = parser.parse_args()

torch.manual_seed(args.seed)

import logging
logging.basicConfig(filename=args.name + '.log',level=logging.DEBUG)

logging.info('Starting logging for {}'.format(args.name))

# Training data
patients = get_list_of_patients('brats_data_preprocessed/Brats{}TrainingData'.format(str(args.brats_train_year)))
patients = patients[0:args.training_max]
print(f"The number of training patients: {len(patients)}")

batch_size = args.batch_size
patch_size = [args.patch_depth, args.patch_width, args.patch_height]

if args.num_channels == 3:
    in_channels = ['t1c', 't2', 'flair']

elif args.num_channels == 4:
    in_channels = ['t1', 't1c', 't2', 'flair']


#%%
# num_splits=5 means 1/5th is validation data!
patients_train, patients_val = get_split_deterministic(patients, fold=0, num_splits=5, random_state=args.seed)

if not args.use_validation:
    patients_train = patients

#%%
train_dl = BRATSDataLoader(
    patients_train,
    batch_size=batch_size,
    patch_size=patch_size,
    in_channels=in_channels
)

val_dl = BRATSDataLoader(
    patients_val,
    batch_size=batch_size,
    patch_size=patch_size,
    in_channels=in_channels
)
#%%
#tr_transforms = get_train_transform(patch_size)
tr_transforms = get_train_transform(patch_size, noise="Riccian")
#tr_transforms = get_train_transform_aggro(patch_size)



# finally we can create multithreaded transforms that we can actually use for training
# we don't pin memory here because this is pytorch specific.
tr_gen = MultiThreadedAugmenter(train_dl, tr_transforms, num_processes=4, # num_processes=4
                                num_cached_per_queue=3,
                                seeds=None, pin_memory=False)
# we need less processes for vlaidation because we dont apply transformations
val_gen = MultiThreadedAugmenter(val_dl, None,
                                 num_processes=max(1, 4 // 2), # num_processes=max(1, 4 // 2)
                                 num_cached_per_queue=1,
                                 seeds=None,
                                 pin_memory=False)

#%%
tr_gen.restart()
val_gen.restart()

#%%
if args.multi_class:
    num_classes = 4
else:
    num_classes = 1


if args.num_channels == 3:
    net_3d = AlbuNet3D34(num_classes=num_classes, pretrained=args.pretrained, is_deconv=True)

elif args.num_channels == 4:
    net_3d = AlbuNet3D34_4channels(num_classes=num_classes, pretrained=args.pretrained, is_deconv=True,updated=True)


#%%
loss_fn = GeneralizedDiceLoss() if args.multi_class else SimpleDiceLoss()
metric = dice_multi_class if args.multi_class else dice

# print(f"Training batch size is: {args.training_batch_size}")
# print(f"Validation batch size is: {args.validation_batch_size}")
print(f"Training for {args.epochs} epochs")

model_trainer = ModelTrainer(args.name, net_3d, tr_gen, val_gen, loss_fn, metric,
                             lr=args.learning_rate, epochs=args.epochs,
                             num_batches_per_epoch=args.training_batch_size, num_validation_batches_per_epoch=args.validation_batch_size,
                             use_gpu=args.use_gpu, multi_class=args.multi_class)    


model_trainer.run()