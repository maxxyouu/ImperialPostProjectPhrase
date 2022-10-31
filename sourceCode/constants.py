import torch
import os

USE_GPU = True
# training device
DTYPE = torch.float32
DEVICE = torch.device('cpu')
if USE_GPU and torch.cuda.is_available():
    DEVICE = torch.device('cuda:0')

# WORK_ENV = 'COLAB'
WORK_ENV = 'LOCAL'

if WORK_ENV == 'COLAB':
    PASCAL_DATA_PATH = '/content/drive/MyDrive/'
    IMGNET_DATA_PATH = '/content/drive/MyDrive/'
    SAVED_MODEL_PATH = '/content/drive/MyDrive/VOC2012_trained_models'
    STORAGE_PATH = '/content/drive/MyDrive/'
else:
    PASCAL_DATA_PATH = os.path.join(os.getcwd(), 'data')
    IMGNET_DATA_PATH = os.path.join(os.getcwd(), 'data/ImageNet2012')
    SAVED_MODEL_PATH = os.path.join(os.getcwd(), 'VOC2012_trained_models')
    STORAGE_PATH = os.getcwd()

NUM_CLASSES = 20
SEED = 90

# PASCAL VOC2012
PASCAL_VOC2012 = 'pascalvoc'
CENTRE_CROP_SIZE = 256
PASCAL_DATA_MEAN_R = 0.4611362580203514
PASCAL_DATA_MEAN_G = 0.43324134022359007
PASCAL_DATA_MEAN_B = 0.3998658574846423
PASCAL_DATA_STD_R = 0.2730704311018092
PASCAL_DATA_STD_G = 0.2672517785770057
PASCAL_DATA_STD_B = 0.2776715237923267


# IMAGENET2012
IMGNET2012 = 'imagenet2012'
IMGNET_CENTRE_CROP_SIZE = 224
IMGNET_DATA_MEAN_R = 0.485
IMGNET_DATA_MEAN_G = 0.456
IMGNET_DATA_MEAN_B = 0.406
IMGNET_DATA_STD_R = 0.229
IMGNET_DATA_STD_G = 0.224
IMGNET_DATA_STD_B =  0.225