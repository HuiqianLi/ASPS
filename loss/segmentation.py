import torch
import torch.nn.functional as F
import torch.nn as nn

def dice_loss(pred, target, smooth=1):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice