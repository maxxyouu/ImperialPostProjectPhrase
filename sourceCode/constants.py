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
    SAVED_MODEL_PATH = '/content/drive/MyDrive/VOC2012_trained_models'
else:
    PASCAL_DATA_PATH = os.path.join(os.getcwd(), 'data')
    SAVED_MODEL_PATH = os.path.join(os.getcwd(), 'VOC2012_trained_models')

NUM_CLASSES = 20
CENTRE_CROP_SIZE = 256

DATA_MEAN_R = 0.4611362580203514
DATA_MEAN_G = 0.43324134022359007
DATA_MEAN_B = 0.3998658574846423
DATA_STD_R = 0.2730704311018092
DATA_STD_G = 0.2672517785770057
DATA_STD_B = 0.2776715237923267