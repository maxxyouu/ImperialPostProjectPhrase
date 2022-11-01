from copy import deepcopy
import constants
import os
import tqdm
import cv2
import logging


from skresnet import skresnext50_32x4d
from resnet import resnet50
from layers import *
from utils import *

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
import torch.nn.functional as F

def max_min_lrp_normalize(Ac):
    Ac_shape = Ac.shape
    AA = Ac.view(Ac.size(0), -1)
    AA -= AA.min(1, keepdim=True)[0]
    AA /= AA.max(1, keepdim=True)[0]
    scaled_ac = AA.view(Ac_shape)
    return scaled_ac

def tensor2image(x, size=constants.IMGNET_CENTRE_CROP_SIZE):
    x = max_min_lrp_normalize(x).detach().cpu().numpy()
    x = deepcopy(x)
    results = np.empty((x.shape[0], size, size))
    for i in range(x.shape[0]):
        results[i, :] = cv2.resize(np.transpose(x[i,:], (1, 2, 0)).reshape((x.shape[-1], x.shape[-1])), (size, size))

    return results


def segmentation_evaluation():
    pass

def model_metric_evaluation(args, dataset, loader, model, ):
    '''
    log the A.D, I.C, and Confidence Drop scores
    '''
    pass

def confidence_drop_evaluation():
    pass

def average_increase_evaluation():
    pass

def average_drop_evaluation():
    pass