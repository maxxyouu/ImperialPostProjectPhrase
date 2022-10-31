# download pascal data

# DL-related
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy
from utils import collate_function

# others
import os
# fully works
val_set = torchvision.datasets.ImageNet(
    root=os.path.join(os.getcwd(), 'data/ImageNet2012')
    ,split='val'
    ,transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(256)
    ])
)

val_loader = DataLoader(
    val_set
    ,batch_size=2
    ,shuffle=True
)
# sample = next(iter(val_loader))
# print('hello')
