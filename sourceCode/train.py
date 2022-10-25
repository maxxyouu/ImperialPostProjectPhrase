import argparse
import pickle
import os
import numpy as np
from sklearn.metrics import average_precision_score, accuracy_score, f1_score, precision_score, recall_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import timm
from skresnet import skresnext50_32x4d

import constants
from utils import collate_function

########################## PARSE SCRIPT ARGUMENTS STARTS ##########################
my_parser = argparse.ArgumentParser(description='')
my_parser.add_argument('--model',
                        type=str, default='skresnext50_32x4d',
                        help='model to be used for training / testing')
my_parser.add_argument('--batch_size',
                        type=int, default=2,
                        help='batch size to be used for training / testing')             
my_parser.add_argument('--epochs',
                        type=int, default=100,
                        help='training epochs')   
my_parser.add_argument('--earlyStoppingPatience',
                        type=int, default=10,
                        help='early stopping patience to terminate the training process')   
my_parser.add_argument('--learningRate',
                        type=float, default=0.001,
                        help='learning rate for training') 
my_parser.add_argument('--pretrain',
                        type=bool, action=argparse.BooleanOptionalAction,
                        help='whether to use a pretrained model')
my_parser.add_argument('--checkPointLocation',
                        type=str, default='',
                        help='Checkpoint location') 
args = my_parser.parse_args()

print('--model: {}'.format(args.model))
print('--batchSize: {}'.format(args.batch_size))
print('--epochs: {}'.format(args.epochs))
print('--earlyStoppingPatience: {}'.format(args.earlyStoppingPatience))
print('--learningRate: {}'.format(args.learningRate))
print('--pretrain: {}'.format(args.pretrain))
print('--checkPointLocation: {}'.format(args.checkPointLocation))

# Default choice for the script parameters
if args.pretrain is None:
    args.pretrain = True
########################## PARSE SCRIPT ARGUMENTS ENDS ##########################

########################## CREATE MODEL STARTS ##########################

model = timm.create_model(args.model, pretrained=args.pretrain) 
# model = skresnext50_32x4d(pretrained=args.pretrain)
model.fc = nn.Linear(model.fc.in_features, constants.NUM_CLASSES, device=constants.DEVICE, dtype=constants.DTYPE)

########################## DATA STARTS ##########################
train_transforms = transforms.Compose(
    [
        transforms.ToTensor(), 
        transforms.CenterCrop(constants.CENTRE_CROP_SIZE), # transforms.CenterCrop((336, 350)), 230 is the number that has the largest square in a circle
        transforms.RandomHorizontalFlip(),
        transforms.RandomPerspective(), # perspective invarient
        transforms.GaussianBlur(kernel_size=(5, 9)), # meant for distant object highlight
        transforms.RandomRotation((0, 270)), # rotation invarient
        transforms.RandomAutocontrast(0.25),
        transforms.Normalize(
        [constants.DATA_MEAN_R, constants.DATA_MEAN_G, constants.DATA_MEAN_B], 
        [constants.DATA_STD_R,constants.DATA_STD_G, constants.DATA_STD_B])
    ]
)
val_test_transforms = transforms.Compose([
    transforms.ToTensor(), 
    transforms.CenterCrop(constants.CENTRE_CROP_SIZE),
    transforms.Normalize(
    [constants.DATA_MEAN_R, constants.DATA_MEAN_G, constants.DATA_MEAN_B], 
    [constants.DATA_STD_R,constants.DATA_STD_G, constants.DATA_STD_B])
])

train_set = torchvision.datasets.VOCDetection(
    root=constants.PASCAL_DATA_PATH
    ,year='2012'
    ,image_set='train'
    ,download=False
    ,transform=train_transforms
)

val_set = torchvision.datasets.VOCDetection(
    root=constants.PASCAL_DATA_PATH
    ,year='2012'
    ,image_set='val'
    ,download=False
    ,transform=val_test_transforms
)

train_loader = DataLoader(
    train_set
    ,batch_size=args.batch_size
    ,shuffle=True
    ,collate_fn=collate_function
)

val_loader = DataLoader(
    train_set
    ,batch_size=args.batch_size
    ,shuffle=True
    ,collate_fn=collate_function
)

########################## DATA ENDS ##########################

########################## TRAIN LOOP STARTS ##########################
def check_accuracy(model, loader, threshold=0.5):
    model.eval()
    avgs = []
    with torch.no_grad():

        for x, y in loader:
            x = x.to(device=constants.DEVICE, dtype=constants.DTYPE)  # move to device
            y = y.to(device=constants.DEVICE, dtype=torch.long)
            logits = model(x)
            category_probs = F.sigmoid(logits)

            # bookkeepings
            category_probs[category_probs > threshold] = 1
            category_probs[category_probs <= threshold] = 0
            acc = accuracy_score(y.cpu().tolist(), category_probs.cpu().tolist())
            avgs.append(acc)

    return sum(avgs)/len(avgs)


def confusion_scores(model, loader, threshold=.5):
    model.eval()
    precisions, recalls, f1s = [], [], []
    with torch.no_grad():

        for x, y in loader:
            x = x.to(device=constants.DEVICE, dtype=constants.DTYPE)  # move to device
            y = y.to(device=constants.DEVICE, dtype=torch.long)
            logits = model(x)
            category_probs = F.sigmoid(logits)

            # bookkeepings
            category_probs[category_probs > threshold] = 1
            category_probs[category_probs <= threshold] = 0
            precision = precision_score(y.cpu().tolist(), category_probs.cpu().tolist(), average='macro')
            recall = recall_score(y.cpu().tolist(), category_probs.cpu().tolist(), average='macro')
            f1 = f1_score(y.cpu().tolist(), category_probs.cpu().tolist(), average='macro')

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

    return sum(precisions)/len(precisions), sum(recalls)/len(recalls), sum(f1s)/len(f1s)


model = model.to(device=constants.DEVICE)  # move the model parameters to CPU/GPU
optimizer = optim.Adamax(model.parameters(), lr=args.learningRate, weight_decay=1e-8)
criterion = nn.BCEWithLogitsLoss()

patience, optimal_val_loss = args.earlyStoppingPatience, np.inf
train_losses, val_losses = [], []
for e in range(args.epochs):
    per_epoch_train_loss = []
    for t, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        model.train()
        x = x.to(device=constants.DEVICE, dtype=constants.DTYPE)  # move to device, e.g. GPU
        y = y.to(device=constants.DEVICE, dtype=constants.DTYPE)
        logits = model(x)
        loss = criterion(logits, y)
        
        # post processing
        per_epoch_train_loss.append(loss.cpu().detach().numpy())
        loss.backward()
        optimizer.step()
    
    # check validation accuracy
    precision, recall, f1 = confusion_scores(model, val_loader)
    val_acc, val_loss = check_accuracy(model, val_loader)

    # record the statistics
    val_losses.append(val_loss.cpu().detach().numpy())
    train_losses.append(sum(per_epoch_train_loss) / len(per_epoch_train_loss))
    print('Epoch: {}, val_loss {}, val_acc {}, precision {}, recall {}, f1 {}'.format(e, val_loss, val_acc, precision, recall, f1))
    
    # early stopping mechanism
    if val_loss < optimal_val_loss:
        print('Saving model')
        model_dest = os.path.join(args.checkPointLocation, '{}.pt'.format(args.model))
        torch.save(model.state_dict(), model_dest)
        optimal_val_loss = val_loss
        patience = args.earlyStoppingPatience
    else:
        patience -= 1
    
    # stop training when epoch acc no longer improve for consecutive {patience} epochs
    if patience <= 0:
        break
########################## TRAIN LOOP ENDS ##########################













