# download pascal data

# DL-related
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

# others
import os

train_set = torchvision.datasets.CocoDetection(
    root='./data'
    ,annFile='./cocoAnno'
    ,transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

train_loader = DataLoader(train_set
    ,batch_size=20
    ,shuffle=True
)
