
import argparse
from inspect import getsource

import constants
import os
import tqdm
import cv2
import logging


# from skresnet import skresnext50_32x4d
from resnet import resnet50
from layers import *
from utils import *
from evaluationUtils import *

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler


########################## PARSE SCRIPT ARGUMENTS STARTS ##########################

my_parser = argparse.ArgumentParser(description='')
default_model_name = 'resnet50'
my_parser.add_argument('--model',
                        type=str, default=default_model_name,
                        help='model to be used for training / testing') 
my_parser.add_argument('--pickle_name',
                        type=str, default=default_model_name+'_pretrain.pt',
                        help='pickel name for weight loading') 
my_parser.add_argument('--state_dict_path',
                        type=str, default=os.path.join(constants.SAVED_MODEL_PATH, default_model_name+'_pretrained.pt'),
                        help='location of the weight pickle file') 
my_parser.add_argument('--target_layer',
                        type=str, default='layer2',
                        help='cam layer for explanation') 
my_parser.add_argument('--batch_size',
                        type=int, default=3,
                        help='batch size to be used for training / testing') 
my_parser.add_argument('--XRelevanceCAM',
                        type=bool, action=argparse.BooleanOptionalAction,
                        help='use xrelevance cam')
my_parser.add_argument('--correctPredictionsOnly',
                        type=bool, action=argparse.BooleanOptionalAction,
                        help='only evaluate on the correct predictions only')
my_parser.add_argument('--alpha',
                        type=int, default=1,
                        help='alpha in the propagation rule')  
my_parser.add_argument('--dataset',
                        type=str, default=constants.IMGNET2012,
                        help='dataset to be tested')
my_parser.add_argument('--model_metric',
                        type=str, default='AD',
                        help='AD, IC, XAD')
args = my_parser.parse_args()

print('--model: {}'.format(args.model))
print('--pickle_name: {}'.format(args.pickle_name))
print('--state_dict_path: {}'.format(args.state_dict_path))
print('--target_layer: {}'.format(args.target_layer))
print('--batch_size: {}'.format(args.batch_size))

if not args.XRelevanceCAM: # and constants.WORK_ENV == 'LOCAL': # for debug purposes
    args.XRelevanceCAM = False
print('--XRelevanceCAM: {}'.format(args.XRelevanceCAM))

if not args.correctPredictionsOnly:
    args.correctPredictionsOnly = False
print('--correctPredictionsOnly: {}'.format(args.correctPredictionsOnly))

print('--alpha: {}'.format(args.alpha))
print('--dataset: {}'.format(args.dataset))

if len(args.model_metric) > 0:
    print('--model_metric: {}'.format(args.model_metric))
else:
    args.model_metric = None

CAM_NAME = 'RelevanceCAM' if not args.XRelevanceCAM else 'XRelevanceCAM'
########################## PARSE SCRIPT ARGUMENTS ENDS ##########################

########################## LOAD THE TRAINED MODEL STARTS ##########################

if args.model == 'resnet50':
    model = resnet50(pretrained=True).eval()
elif args.model == 'skresnext50_32x4d':
    model = skresnext50_32x4d(pretrained=True).eval()

if args.dataset == constants.PASCAL_VOC2012:
    model.num_classes = constants.PASCAL_NUM_CLASSES #NOTE required to do CLRP and SGLRP
    model.fc = Linear(model.fc.in_features, constants.PASCAL_NUM_CLASSES, device=constants.DEVICE)
    model.load_state_dict(torch.load(args.state_dict_path, map_location=constants.DEVICE))

model.to(constants.DEVICE)
print('Model successfully loaded')

if args.target_layer == 'layer2':
    target_layer = model.layer2
elif args.target_layer == 'layer3':
    target_layer = model.layer3
elif args.target_layer == 'layer4':
    target_layer = model.layer4
else:
    target_layer = model.layer1

value = dict()
def forward_hook(module, input, output):
    value['activations'] = output
def backward_hook(module, input, output):
    value['gradients'] = output[0]

# create folder to store the segmented image if not exists
origin_dest = os.path.join(constants.STORAGE_PATH,
                    'IMGNET2012HeatMaps' if args.dataset == constants.IMGNET2012 else 'VOC2012HeatMaps')
if not os.path.exists(origin_dest):
    os.makedirs(origin_dest)

########################## LOAD THE TRAINED MODEL ENDS ##########################

########################## DATA STARTS ##########################
# TODO: SELECT A SUBSET OF DATA FOR IMAGENET AND USING VOC2012 SEGMENT FOR VOC DATASET
# TODO: RETRAIN THE NETWORK BY INCLUDING THE TRAINING-SEGMENTED IMAGES AND EVALUTE ON EITHER THE VAL-SEGMENTATION OR VAL + VAL-SEG IMAGES
# TODO: RETRAIN THE NETWORK BY INCLUDING ONLY THE TRAINING-SEGMENTED IMAGES AND EVALUTE ON ONLY THE VAL-SEGMENTATION OR VAL-SEG IMAGES
# TODO: CHECK THE IMPLEMENTATION OF THE ALEXNET: resize and crop
    # https://github.com/pytorch/examples/blob/97304e232807082c2e7b54c597615dc0ad8f6173/imagenet/main.py#L197-L198

if args.dataset == constants.IMGNET2012:
    val_set = torchvision.datasets.ImageNet(
        root=constants.IMGNET_DATA_PATH
        ,split='val'
        ,transform=transforms.Compose([
            transforms.Resize((constants.IMGNET_CENTRE_CROP_SIZE, constants.IMGNET_CENTRE_CROP_SIZE))
            ,transforms.ToTensor()
            ,transforms.Normalize(
                [constants.IMGNET_DATA_MEAN_R, constants.IMGNET_DATA_MEAN_G, constants.IMGNET_DATA_MEAN_B],
                [constants.IMGNET_DATA_STD_R, constants.IMGNET_DATA_STD_G, constants.IMGNET_DATA_STD_B]
            )])
        )
    # 5000 images for evaluation
    train_size = int(len(val_set)*0.9)
    val_size = len(val_set) - train_size

    # indices of the sampled images
    _, val_set = torch.utils.data.random_split(val_set, [train_size, val_size], generator=torch.Generator().manual_seed(constants.SEED))

    inplace_normalize = transforms.Normalize([constants.IMGNET_DATA_MEAN_R, constants.IMGNET_DATA_MEAN_G, constants.IMGNET_DATA_MEAN_B],
                                             [constants.IMGNET_DATA_STD_R, constants.IMGNET_DATA_STD_G, constants.IMGNET_DATA_STD_B], inplace=True)
    filenames = val_set.dataset.imgs

    sequentialSampler = SequentialSampler(val_set)
    val_loader = DataLoader(
        val_set
        ,batch_size=args.batch_size
        # ,shuffle=True
        ,sampler=sequentialSampler
        # ,collate_fn=collate_function # for voc2012 only
    )
    indices = val_set.indices

elif args.dataset == constants.PASCAL_VOC2012:
    trainval_set = torchvision.datasets.VOCDetection(
        root=constants.PASCAL_DATA_PATH
        ,year='2012'
        ,image_set='trainval'
        ,download=False
        ,transform=transforms.Compose([
            transforms.Resize((constants.PASCAL_CENTRE_CROP_SIZE,constants.PASCAL_CENTRE_CROP_SIZE))
            ,transforms.ToTensor()
            ,transforms.Normalize(
                [constants.PASCAL_DATA_MEAN_R, constants.PASCAL_DATA_MEAN_G, constants.PASCAL_DATA_MEAN_B], 
                [constants.PASCAL_DATA_STD_R, constants.PASCAL_DATA_STD_G, constants.PASCAL_DATA_STD_B])
            ])
        
        #,target_transform=encode_segmentation_labels # for segmentation only
    )
    # only a subset of data are used
    train_size = int(len(trainval_set)*0.8)
    val_size = len(trainval_set) - train_size

    # indices of the sampled images from the validation set
    _, val_set = torch.utils.data.random_split(trainval_set, [train_size, val_size], generator=torch.Generator().manual_seed(constants.SEED))

    inplace_normalize = transforms.Normalize([constants.PASCAL_DATA_MEAN_R, constants.PASCAL_DATA_MEAN_G, constants.PASCAL_DATA_MEAN_B], 
                                             [constants.PASCAL_DATA_STD_R, constants.PASCAL_DATA_STD_G, constants.PASCAL_DATA_STD_B], inplace=True)
    filenames = val_set.dataset.images
    indices = val_set.indices


    # val_set = torchvision.datasets.VOCSegmentation(
    #     root=constants.PASCAL_DATA_PATH
    #     ,year='2012'
    #     ,image_set='val'
    #     ,download=False
    #     ,transform=transforms.Compose([
    #         transforms.Resize((constants.PASCAL_CENTRE_CROP_SIZE,constants.PASCAL_CENTRE_CROP_SIZE))
    #         ,transforms.ToTensor()
    #         ,transforms.Normalize(
    #             [constants.PASCAL_DATA_MEAN_R, constants.PASCAL_DATA_MEAN_G, constants.PASCAL_DATA_MEAN_B], 
    #             [constants.PASCAL_DATA_STD_R, constants.PASCAL_DATA_STD_G, constants.PASCAL_DATA_STD_B])
    #         ])
    #     #,target_transform=encode_segmentation_labels # for segmentation only
    # )
    # # only a subset of data are used
    # train_size = int(len(trainval_set)*0.8)
    # val_size = len(trainval_set) - train_size
    # _, val_set = torch.utils.data.random_split(trainval_set, [train_size, val_size], generator=torch.Generator().manual_seed(constants.SEED))


    # inplace_normalize = transforms.Normalize([constants.PASCAL_DATA_MEAN_R, constants.PASCAL_DATA_MEAN_G, constants.PASCAL_DATA_MEAN_B], 
    #                                          [constants.PASCAL_DATA_STD_R, constants.PASCAL_DATA_STD_G, constants.PASCAL_DATA_STD_B], inplace=True)
    # filenames = val_set.images

    sequentialSampler = SequentialSampler(val_set)
    val_loader = DataLoader(
        val_set
        ,batch_size=args.batch_size
        # ,shuffle=True
        ,sampler=sequentialSampler
        ,collate_fn=collate_function #collate_function_voc2012_segs # for voc2012
    )

########################## DATA ENDS ##########################

########################## EVALUATION STARTS ##########################
print('Evaluation Begin')
STARTING_INDEX = 0

if args.model_metric == 'AD':
    metric = Average_drop_score()
    explanation_map_extractor = hard_threshold_explanation_map
elif args.model_metric == 'IC':
    metric = Increase_confidence_score()
    explanation_map_extractor = hard_threshold_explanation_map
elif args.model_metric == 'XAD':
    metric = Axiom_style_confidence_drop_logger()
    explanation_map_extractor = axiom_paper_average_drop_explanation_map


# subset_dataset.indices
for x, y in val_loader:

    forward_handler = target_layer.register_forward_hook(forward_hook)
    backward_handler = target_layer.register_full_backward_hook(backward_hook)
    x = x.to(device=constants.DEVICE, dtype=constants.DTYPE)  # move to device
    y = y.to(device=constants.DEVICE, dtype=constants.DTYPE)

    print('--------- Forward Passing ------------')
    # use the label to propagate NOTE: another case
    targets = [None] # torch.argmax(y, dim=1) for voc2012
    internal_R_cams, output = model(x, args.target_layer, targets, axiomMode=True if args.XRelevanceCAM else False)
    r_cams = internal_R_cams[0] # for each image in a batch
    r_cams = tensor2image(r_cams)

    predictions = torch.argmax(output, dim=1)

    # denormalize the image NOTE: must be placed after forward passing
    x = denorm(x)
    print('--------- Generating relevance-cam Heatmap')
    for i in range(x.shape[0]):   

        #ignore the wrong prediction
        # if predictions[i] != y[i]:
        #     continue

        # _filename = filenames[indices[STARTING_INDEX + i]] # use the indices to get the filename for voc2012
        _filename, label = filenames[indices[STARTING_INDEX + i]] # use the indices to get the filename

        dest = os.path.join(origin_dest, '{}/{}'.format(args.model, _filename[:-4]))
        img = get_source_img(_filename)

        # save the original image in parallel
        if not os.path.exists(dest):
            os.makedirs(dest)
            plt.axis('off')
            plt.imshow(img)
            plt.savefig(os.path.join(dest, 'original.jpeg'), bbox_inches='tight')
    
        plt.ioff()
        logger = logging.getLogger()
        old_level = logger.level
        logger.setLevel(100)

        # save the saliency map of the image
        r_cam = r_cams[i,:]
        mask = plt.imshow(r_cam, cmap='seismic')
        overlayed_image = plt.imshow(img, alpha=.5)
        plt.axis('off')
        plt.savefig(os.path.join(dest, '{}_{}_{}_seismic.jpeg'.format(CAM_NAME, args.target_layer, predictions[i])), bbox_inches='tight')

        # save the segmentation of the image
        segmented_image = img*threshold(r_cam)[...,np.newaxis]
        segmented_image = plt.imshow(segmented_image)
        plt.axis('off')
        plt.savefig(os.path.join(dest, '{}_{}_{}_segmentation.jpeg'.format(CAM_NAME, args.target_layer, predictions[i])), bbox_inches='tight')
        plt.close()

        logger.setLevel(old_level)

        # update the sequential index for next iterations
        forward_handler.remove()
        backward_handler.remove()
    
    #BOOKING
    STARTING_INDEX += x.shape[0]

# model_metric_evaluation(args, val_set, val_loader, model, inplace_normalize, metrics_logger=metric, xmap_extractor=explanation_map_extractor)