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
# NOTE: the data folder name requires VOCdevkit/VOC2012, required
trainVal_set = torchvision.datasets.VOCDetection(
    root=os.path.join(os.getcwd(), 'data')
    ,year='2012'
    ,image_set='trainval'
    ,download=False
    ,transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(256)
    ])
)


trainval_loader = DataLoader(trainVal_set
    ,batch_size=3
    ,shuffle=True
    ,collate_fn=collate_function # this function is required to return same-length labels and inputs
)
# sample, label = next(iter(trainval_loader)) # each object inside the label['annotation']['object'] list
x2s, x3s = [], [] # x is at index 2, y is at index 3
r, g, b = [], [], [] # r = tensor[:,0,:], g = tensor[:,1,:], b = tensor[:,2,:]
rv, gv, bv = 0, 0, 0
for image, labels in trainval_loader:
    assert(list(image.shape) == [1, 3, 256, 256])
    # x2, x3 = image.shape[-2], image.shape[-1]
    # x2s.append(x2)
    # x3s.append(x3)
    rgb = torch.sum(image, dim=(2,3))
    r.append(rgb[:,0].item())
    g.append(rgb[:,1].item())
    b.append(rgb[:,2].item())

    # rv += ()


# print(sum(r)/(len(r)*256*256))
# print(sum(g)/(len(g)*256*256))
# print(sum(b)/(len(b)*256*256))

import statistics
print(statistics.stdev(rv))
print(statistics.stdev(gv))
print(statistics.stdev(bv))



# print(min(x2s), min(x3s))