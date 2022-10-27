import argparse
import pickle
import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import timm
import sklearn
# from skresnet import skresnext50_32x4d

import constants
# from utils import encode_labels
from utils import collate_function, get_metric_scores, encode_labels

########################## PARSE SCRIPT ARGUMENTS STARTS ##########################
my_parser = argparse.ArgumentParser(description='')
my_parser.add_argument('--model',
                        type=str, default='skresnext50_32x4d',
                        help='model to be used for training / testing')
my_parser.add_argument('--batch_size',
                        type=int, default=2,
                        help='batch size to be used for training / testing')             
my_parser.add_argument('--epochs',
                        type=int, default=3,
                        help='training epochs')   
my_parser.add_argument('--earlyStoppingPatience',
                        type=int, default=20,
                        help='early stopping patience to terminate the training process')   
my_parser.add_argument('--learningRate',
                        type=float, default=0.001,
                        help='learning rate for training') 
my_parser.add_argument('--pretrain',
                        type=bool, action=argparse.BooleanOptionalAction,
                        help='whether to use a pretrained model')
my_parser.add_argument('--checkPointLocation',
                        type=str, default=constants.SAVED_MODEL_PATH,
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

if not os.path.exists(args.checkPointLocation):
    os.makedirs(args.checkPointLocation)

rng_seed = 90
torch.manual_seed(rng_seed)
########################## PARSE SCRIPT ARGUMENTS ENDS ##########################

########################## CREATE MODEL STARTS ##########################

model = timm.create_model(args.model, pretrained=args.pretrain) 
model.fc = nn.Linear(model.fc.in_features, constants.NUM_CLASSES, device=constants.DEVICE, dtype=constants.DTYPE)

# saved_model = torch.load('./VOC2012_trained_models/skresnext50_32x4d_pretrained.pt', map_location=constants.DEVICE)
# model.load_state_dict(saved_model)

########################## DATA STARTS ##########################
train_transforms = transforms.Compose(
    [
        transforms.ToTensor(), 
        transforms.CenterCrop(constants.CENTRE_CROP_SIZE), # transforms.CenterCrop((336, 350)), 230 is the number that has the largest square in a circle
        transforms.RandomHorizontalFlip(),
        transforms.RandomPerspective(), # perspective invarient
        transforms.RandomRotation((0, 270)), # rotation invarient
        transforms.RandomAutocontrast(0.25),
        transforms.Normalize(
        [constants.DATA_MEAN_R, constants.DATA_MEAN_G, constants.DATA_MEAN_B], 
        [constants.DATA_STD_R,constants.DATA_STD_G, constants.DATA_STD_B])
    ]
)
test_transforms = transforms.Compose([
    transforms.ToTensor(), 
    transforms.CenterCrop(constants.CENTRE_CROP_SIZE),
    transforms.Normalize(
    [constants.DATA_MEAN_R, constants.DATA_MEAN_G, constants.DATA_MEAN_B], 
    [constants.DATA_STD_R,constants.DATA_STD_G, constants.DATA_STD_B])
])

trainval_set = torchvision.datasets.VOCDetection(
    root=constants.PASCAL_DATA_PATH
    ,year='2012'
    ,image_set='trainval'
    ,download=False
    ,transform=train_transforms
    ,target_transform=encode_labels
)

train_size = int(len(trainval_set)*0.8)
val_size = len(trainval_set) - train_size
train_set, val_set = torch.utils.data.random_split(trainval_set, [train_size, val_size], generator=torch.Generator().manual_seed(rng_seed))

train_loader = DataLoader(
    train_set
    ,batch_size=args.batch_size
    ,shuffle=True
)

val_loader = DataLoader(
    val_set
    ,batch_size=args.batch_size
    ,shuffle=True
)

########################## DATA ENDS ##########################

########################## TRAIN LOOP STARTS ##########################


def confusion_scores(model, loader, criterion):
    model.eval()
    num_samples = 0

    running_avg_ap = 0.0 # is label 1 correctly idenfified
    running_precision = 0.0
    running_recall = 0.0
    running_f1 = 0.0
    val_loss = 0.0

    with torch.no_grad():

        for x, y in tqdm(loader):
            x = x.to(device=constants.DEVICE, dtype=constants.DTYPE)  # move to device
            y = y.to(device=constants.DEVICE, dtype=constants.DTYPE)
            logits = model(x)

            avg_ap, p, r, f = get_metric_scores(y.cpu().numpy(), torch.sigmoid(logits).cpu().numpy())
            running_avg_ap += avg_ap
            running_precision += p
            running_recall += r
            running_f1 += f
            val_loss += criterion(logits, y).cpu().item()
            num_samples += x.shape[0]

    return (running_avg_ap/num_samples,
           running_precision/num_samples,
           running_recall/num_samples, 
           running_f1/num_samples,
           val_loss/num_samples)

model = model.to(device=constants.DEVICE)  # move the model parameters to CPU/GPU
optimizer = optim.Adamax(model.parameters(), lr=args.learningRate, weight_decay=1e-8)
criterion = nn.BCEWithLogitsLoss()

print('Training Began.')
patience, optimal_val_loss = args.earlyStoppingPatience, np.inf
train_losses, val_losses = [], []
for e in range(args.epochs):
    per_epoch_train_loss = []
    for x, y in tqdm(train_loader):
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
    avg_ap, precision, recall, f1, val_loss = confusion_scores(model, val_loader, criterion)

    # record the statistics
    val_losses.append(val_loss)
    train_losses.append(sum(per_epoch_train_loss) / len(per_epoch_train_loss))
    # print('Epoch: {}, val_loss {}, avg_precision {}, precision {}, recall {}, f1 {}'.format(e, val_loss, avg_ap, precision, recall, f1))
    print('Epoch: {}, val_loss {}, avg_precision {}'.format(e, val_loss, avg_ap))
    
    # early stopping mechanism
    if val_loss < optimal_val_loss:
        print('Epoch {} ------ Saving model'.format(e))
        model_dest = os.path.join(args.checkPointLocation, '{}{}.pt'.format(args.model, '_pretrained' if args.pretrain else ''))
        torch.save(model.state_dict(), model_dest)
        optimal_val_loss = val_loss
        patience = args.earlyStoppingPatience
    else:
        patience -= 1
    
    # stop training when epoch acc no longer improve for consecutive {patience} epochs
    if patience <= 0:
        break
########################## TRAIN LOOP ENDS ##########################

print('Training Completed.')