import torch
import numpy as np
import math

#%%
def get_region(labels, region='tumor_core'):
    if region=='tumor_core':
        return ((labels == 1) | (labels == 3))
    elif region=='edema':
        return ((labels == 1) | (labels == 2) | (labels == 3))
    elif region=='enhancing':
        return (labels == 3)


#%%
def dice(outputs, targets, label='tumor_core'):

    # try without sigmoid
    # outputs = F.sigmoid(outputs)
    outputs = (outputs>0).float()
    smooth = 1e-15

    targets = get_region(targets, label).float()
    union_fg = (outputs+targets).sum() + smooth
    intersection_fg = (outputs*targets).sum()

    dice = 2 * intersection_fg / union_fg

    return dice

#%%
def dice_multi_class(outputs, targets):
    outputs = outputs.argmax(dim=1, keepdim=False)
    targets = targets.argmax(dim=1, keepdim=False)
    smooth = 1e-15

    dices = []

    for region in ['edema', 'tumor_core', 'enhancing']:
        output_region = get_region(outputs, region).float()
        target_region = get_region(targets, region).float()

        union_fg = (output_region+target_region).sum() + smooth
        intersection_fg = (output_region*target_region).sum()

        dice = 2 * intersection_fg / union_fg
        dices.append(dice)

    return dices


#%%
# Differentiable version of the dice metric
class SimpleDiceLoss():
    def __call__(self, outputs, targets, label='tumor_core'):

        # try without sigmoid
        # outputs = F.sigmoid(outputs)
        outputs = torch.sigmoid(outputs)
        # outputs = (outputs>0).float()
        smooth = 1e-15
        
        targets = get_region(targets, label).float()
        union_fg = (outputs+targets).sum() + smooth
        intersection_fg = (outputs*targets).sum()
        
        dice = 2 * intersection_fg / union_fg

        return 1 - dice

#%%
# Differentiable version of the dice metric
class GeneralizedDiceLoss():
    def __call__(self, outputs, targets):

        outputs = torch.nn.functional.softmax(outputs, dim=1)
        smooth = 1e-15

        num_channels = outputs.size(1)

        total_dice = 0

        for ch in range(num_channels):
            union_fg = (outputs[:, ch]+targets[:, ch]).sum() + smooth
            intersection_fg = (outputs[:, ch]*targets[:, ch]).sum()
            total_dice += 2 * intersection_fg / union_fg

        return 1 - total_dice / num_channels

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