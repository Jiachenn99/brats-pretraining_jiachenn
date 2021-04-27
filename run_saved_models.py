
#%%
from brats_data_loader import get_list_of_patients, get_train_transform, iterate_through_patients, BRATSDataLoader
from train_test_function import ModelTrainer
from models import AlbuNet3D34, AlbuNet3D34_4channels
from loss import *

from batchgenerators.utilities.data_splitting import get_split_deterministic
from batchgenerators.dataloading import MultiThreadedAugmenter
from batchgenerators.augmentations.utils import pad_nd_image
from batchgenerators.augmentations.utils import center_crop_3D_image

#%%
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"

#%%
# videos.py
import argparse
parser = argparse.ArgumentParser(description='Run saved models')
parser.add_argument('-model_name', type=str, help='Name of the Model')
parser.add_argument('--patch_depth', type=int, help='Depth of the Input Patch', default=24)
parser.add_argument('--patch_width', type=int, help='Width of the Input Patch', default=128)
parser.add_argument('--patch_height', type=int, help='Height of the Input Patch', default=128)
# parser.add_argument('--epochs_max', type=int, help='Number of epochs of model', default=50)
parser.add_argument('--brats_test_year', type=int, help='BRATS Test Year', default=20)
parser.add_argument('--testing_train_set', type=int, help='Use Training set for test?', default=0)
parser.add_argument('--num_channels', type=int, help='Number of input channels', default=3)
parser.add_argument('--no_gpu', dest='use_gpu', action='store_false', help='Use CPU instead of GPU')
parser.set_defaults(use_gpu=True)
parser.add_argument('--no_multiclass', dest='multi_class', action='store_false', help='Tumor Core Only')
parser.set_defaults(multi_class=True)
args = parser.parse_args()

patch_size = [args.patch_depth, args.patch_width, args.patch_height]

if args.num_channels == 3:
    in_channels = ['t1c', 't2', 'flair']

elif args.num_channels == 4:
    in_channels = ['t1', 't1c', 't2', 'flair']

print(f"Num of channels: {args.num_channels}")

#%% 
# Test data (using validation data of brats20 for testing)

if args.testing_train_set:
    patients_test = get_list_of_patients('brats_data_preprocessed/Brats{}TrainingData'.format(str(args.brats_test_year))) #This is strictly testing purposes
    target_patients = patients_test

else:
    patients_test = get_list_of_patients('brats_data_preprocessed/Brats{}ValidationData'.format(str(args.brats_test_year)))
    target_patients = patients_test

print(f"Test set used is: {args.testing_train_set} (0 for actual test data, 1 for training data")
print(f"The number of testing patients: {len(target_patients)}")


#%%
if args.multi_class:
    num_classes = 4
else:
    num_classes = 1

#%%
try:
    import SimpleITK as sitk
except ImportError:
    raise ImportError("SimpleITK not found")

def save_segmentation_as_nifti(segmentation, metadata, output_file):
    original_shape = metadata['original_shape']
    seg_original_shape = np.zeros(original_shape, dtype=np.uint8)
    nonzero = metadata['nonzero_region']
    seg_original_shape[nonzero[0, 0] : nonzero[0, 1] + 1,
               nonzero[1, 0]: nonzero[1, 1] + 1,
               nonzero[2, 0]: nonzero[2, 1] + 1] = segmentation
    sitk_image = sitk.GetImageFromArray(seg_original_shape)
    sitk_image.SetDirection(metadata['direction'])
    sitk_image.SetOrigin(metadata['origin'])
    # remember to revert spacing back to sitk order again
    sitk_image.SetSpacing(tuple(metadata['spacing'][[2, 1, 0]]))
    # logging.info(output_file)
    sitk.WriteImage(sitk_image, output_file)
#%%
import skimage

def predict_patient_in_patches(patient_data, model, num_channels=3):
    # we pad the patient data in order to fit the patches in it
    patient_data_pd = pad_nd_image(patient_data, [144, 192, 192]) # 24*6, 128+2*32, 128+2*32
    # patches.shape = (1, 1, 6, 3, 3, 1, 3, 24, 128, 128)
    steps = (1,1,args.patch_depth,int(args.patch_width/4),int(args.patch_height/4))
    window_shape = (1, num_channels, args.patch_depth, args.patch_width, args.patch_height)
    patches = skimage.util.view_as_windows(patient_data_pd[:, :num_channels, :, :, :], window_shape=window_shape, step=steps)
    
    # (1, 4, 138, 169, 141)
    target_shape = list(patient_data_pd.shape)
    print(f"BEFORE IF: Target shape in predict patient in patches is: {target_shape}")

    if args.multi_class:
        target_shape[1] = 4
    else:
        target_shape[1] = 1 # only one output channel

    print(f"AFTER IF IF: Target shape in predict patient in patches is: {target_shape}")

    prediction = torch.zeros(*target_shape)
    if args.use_gpu:
        prediction = prediction.cuda()
    
    print(f"Patches shape: {patches.shape}")
    for i in range(patches.shape[2]):
        for j in range(patches.shape[3]):
            for k in range(patches.shape[4]):
                data = torch.from_numpy(patches[0, 0, i, j, k])
                if args.use_gpu:
                    data = data.cuda()
                output = model.forward(data)

                prediction[:, :,
                           i*steps[2]:i*steps[2]+window_shape[2],
                           j*steps[3]:j*steps[3]+window_shape[3],
                           k*steps[4]:k*steps[4]+window_shape[4]] += output
                    
    return prediction

dices = []

#%% Load saved models
model_path = "/cs/home/hfyjc3/brats-pretraining_jiachenn/models/"

if args.num_channels == 3:
    model = AlbuNet3D34(num_classes=4)

elif args.num_channels == 4:
    model =  AlbuNet3D34_4channels(num_classes=4)

model.cuda()

# for epochs in range(0+10, args.epochs_max, 10):
#     model.load_state_dict(torch.load(model_path+f"{args.model_name}_{epochs}"))
#     print(f"Model: {args.model_name}_{epochs}")

#     #%%  Perform prediction and save predictons
#     for idx, (patient_data, meta_data) in enumerate(iterate_through_patients(target_patients, in_channels)): #  + ['seg']
#         print(f"Predicting patient {target_patients[idx].split('/')[-2:][-1]}")
    
#         model.eval()
#         with torch.no_grad():
#             prediction = predict_patient_in_patches(patient_data, model, num_channels=args.num_channels)
        
#         np_prediction = prediction.cpu().detach().numpy()

#         if args.multi_class:
#             np_prediction = np.expand_dims(np.argmax(np_prediction, axis=1), axis=1)
#         else:
#             np_prediction[np_prediction > 0] = 1 # tumor core
#             np_prediction[np_prediction < 0] = 0
    
#         np_cut = center_crop_3D_image(np_prediction[0,0], patient_data.shape[2:])

#         # repair labels
#         np_cut[np_cut == 3] = 4
#         print(f"Shape of np_cut after repair: {np_cut.shape}")
#         print(f"Unique values in np_cut AFTER: {np.unique(np_cut)}")

#         print(f"Seg output shape is: {np_cut.shape}")
#         output_path = '/'.join(target_patients[idx].split('/')[-2:])
#         output_path = os.path.join('segmentation_output', args.model_name+f"_{epochs}", output_path + '.nii.gz')

#         if not os.path.exists(os.path.dirname(output_path)):
#             try:
#                 os.makedirs(os.path.dirname(output_path))
#             except OSError as exc: # Guard against race condition
#                 logging.info('An error occured when trying to create the saving directory!')

#         save_segmentation_as_nifti(np_cut, meta_data, output_path)


#%%
model.load_state_dict(torch.load(model_path+f"{args.model_name}"))
print(f"Model: {args.model_name}")

#%%  Perform prediction and save predictons
for idx, (patient_data, meta_data) in enumerate(iterate_through_patients(target_patients, in_channels)): 
    print(f"Predicting patient {target_patients[idx].split('/')[-2:][-1]}")
    model.eval()
    with torch.no_grad():
        prediction = predict_patient_in_patches(patient_data, model, num_channels=args.num_channels)
    
    np_prediction = prediction.cpu().detach().numpy()

    if args.multi_class:
        np_prediction = np.expand_dims(np.argmax(np_prediction, axis=1), axis=1)
    else:
        np_prediction[np_prediction > 0] = 1 # tumor core
        np_prediction[np_prediction < 0] = 0

    np_cut = center_crop_3D_image(np_prediction[0,0], patient_data.shape[2:])

    # repair labels
    np_cut[np_cut == 3] = 4

    output_path = '/'.join(target_patients[idx].split('/')[-2:])
    output_path = os.path.join('segmentation_output', args.model_name, output_path + '.nii.gz')

    if not os.path.exists(os.path.dirname(output_path)):
        try:
            os.makedirs(os.path.dirname(output_path))
        except OSError as exc: # Guard against race condition
            raise OSError('An error occured when trying to create the saving directory!')

    save_segmentation_as_nifti(np_cut, meta_data, output_path)