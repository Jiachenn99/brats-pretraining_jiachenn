
#%%
from brats_data_loader import get_list_of_patients, get_train_transform, iterate_through_patients, BRATSDataLoader
from train_test_function import ModelTrainer
from jonas_net import AlbuNet3D34

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
parser = argparse.ArgumentParser(description='Run saved models')
parser.add_argument('-model_name', type=str, help='Name of the Model')
parser.add_argument('--batch_size', type=int, help='Batch Size', default=24)
parser.add_argument('--patch_depth', type=int, help='Depth of the Input Patch', default=24)
parser.add_argument('--patch_width', type=int, help='Width of the Input Patch', default=128)
parser.add_argument('--patch_height', type=int, help='Height of the Input Patch', default=128)
parser.add_argument('--brats_test_year', type=int, help='BRATS Test Year', default=20)
parser.add_argument('--no_gpu', dest='use_gpu', action='store_false', help='Use CPU instead of GPU')
parser.set_defaults(use_gpu=True)
parser.add_argument('--no_multiclass', dest='multi_class', action='store_false', help='Tumor Core Only')
parser.set_defaults(multi_class=True)
args = parser.parse_args()

#torch.manual_seed(args.seed)

# Training data
# patients = get_list_of_patients('brats_data_preprocessed/Brats{}TrainingData'.format(str(args.brats_train_year)))
# print(f"The number of training patients: {len(patients)}")
batch_size = args.batch_size
patch_size = [args.patch_depth, args.patch_width, args.patch_height]
in_channels = ['t1c', 't2', 'flair']

#%%
# num_splits=5 means 1/5th is validation data!
# patients_train, patients_val = get_split_deterministic(patients, fold=0, num_splits=5, random_state=args.seed)

#%% 
# Test data (using validation data of brats20, brats18 for testing)
patients_test = get_list_of_patients('brats_data_preprocessed/Brats{}ValidationData'.format(str(args.brats_test_year)))
target_patients = patients_test
print(f"The number of testing patients: {len(target_patients)}")


#%%
def get_region(labels, region='tumor_core'):
    if region=='tumor_core':
        return ((labels == 1) | (labels == 3))
    elif region=='edema':
        return ((labels == 1) | (labels == 2) | (labels == 3))
    elif region=='enhancing':
        return (labels == 3)

#%%
if args.multi_class:
    num_classes = 4
else:
    num_classes = 1

#%%
# try:
import SimpleITK as sitk
# except ImportError:
#     logging.info("You need to have SimpleITK installed to run this example!")
#     raise ImportError("SimpleITK not found")

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
def np_dice(outputs, targets):

    # try without sigmoid
    # outputs = F.sigmoid(outputs)
    outputs = np.float32(outputs)
    smooth = 1e-15

    targets = np.float32((targets == 1) | (targets == 3))
    union_fg = np.sum(outputs+targets) + smooth
    intersection_fg = np.sum(outputs*targets) + smooth

    dice = 2 * intersection_fg / union_fg

    return dice

#%%
def np_dice_multi_class(outputs, targets):
    smooth = 1e-15

    dices = []

    for region in ['edema', 'tumor_core', 'enhancing']:
        output_region = np.float32(get_region(outputs, region))
        target_region = np.float32(get_region(targets, region))

        union_fg = (output_region+target_region).sum() + smooth
        intersection_fg = (output_region*target_region).sum()

        dice = 2 * intersection_fg / union_fg
        dices.append(dice)

    return dices


#%%
import skimage

def predict_patient_in_patches(patient_data, model):
    # we pad the patient data in order to fit the patches in it
    patient_data_pd = pad_nd_image(patient_data, [144, 192, 192]) # 24*6, 128+2*32, 128+2*32
    # patches.shape = (1, 1, 6, 3, 3, 1, 3, 24, 128, 128)
    steps = (1,1,args.patch_depth,int(args.patch_width/4),int(args.patch_height/4))
    window_shape = (1, 3, args.patch_depth, args.patch_width, args.patch_height)
    patches = skimage.util.view_as_windows(patient_data_pd[:, :3, :, :, :], window_shape=window_shape, step=steps)
    
    # (1, 4, 138, 169, 141)
    target_shape = list(patient_data_pd.shape)
    print(f"Target shape in predict patient in patches is: {target_shape}")

    if args.multi_class:
        target_shape[1] = 4
    else:
        target_shape[1] = 1 # only one output channel
    prediction = torch.zeros(*target_shape)
    if args.use_gpu:
        prediction = prediction.cuda()
    
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

#%%
from batchgenerators.augmentations.utils import pad_nd_image
from batchgenerators.augmentations.utils import center_crop_3D_image

dices = []

#%% Load saved models
model_path = "/cs/home/hfyjc3/brats-pretraining_jiachenn/models/"
model = AlbuNet3D34(num_classes=4)
model.cuda()

for epochs in range(0+10, 60+10, 10):
    model.load_state_dict(torch.load(model_path+f"{args.model_name}_{epochs}"))
    print(f"Model: {args.model_name}_{epochs}")

    #%%  Perform prediction and save predictons
    for idx, (patient_data, meta_data) in enumerate(iterate_through_patients(target_patients, in_channels)): #  + ['seg']
        print(f"Predicting patient {target_patients[idx].split('/')[-2:]}")
    
        model.eval()
        with torch.no_grad():
            prediction = predict_patient_in_patches(patient_data, model)
        
        np_prediction = prediction.cpu().detach().numpy()

        if args.multi_class:
            np_prediction = np.expand_dims(np.argmax(np_prediction, axis=1), axis=1)
        else:
            np_prediction[np_prediction > 0] = 1 # tumor core
            np_prediction[np_prediction < 0] = 0
    
        np_cut = center_crop_3D_image(np_prediction[0,0], patient_data.shape[2:])
    
        # if args.multi_class:
        #    dice = np_dice_multi_class(np_cut, patient_data[0,3,:,:,:])
        # else:
        #    dice = np_dice(np_cut, patient_data[0,3,:,:,:])
        # logging.info("{}, {}".format(idx, dice))
        # dices.append(dice)

        # repair labels
        np_cut[np_cut == 3] = 4
        output_path = '/'.join(target_patients[idx].split('/')[-2:])
        output_path = os.path.join('segmentation_output', args.model_name+f"_{epochs}", output_path + '.nii.gz')

        if not os.path.exists(os.path.dirname(output_path)):
            try:
                os.makedirs(os.path.dirname(output_path))
            except OSError as exc: # Guard against race condition
                logging.info('An error occured when trying to create the saving directory!')

        save_segmentation_as_nifti(np_cut, meta_data, output_path)

