from typing import Callable

from matplotlib import image
import constants
import tqdm

from layers import *
from utils import *

def hard_threshold_explanation_map(img, cam):
    """
    used for select which layer of the relevance cam to be used
    """
    explanation_map = img*threshold(cam)
    return explanation_map

def hard_inverse_threshold_explanation_map(img, cam):
    """
    used for select which layer of the relevance cam to be used
    """
    explanation_map = img*threshold(cam, inverse=True)
    return explanation_map

def soft_explanation_map(img, cam):
    """in the grad cam paper
    used for examine the metrics AD, AI
    """
    return img * np.maximum(cam, 0)

def axiom_paper_average_drop_explanation_map(img, cam, inplace_normalize: Callable):
    """

    Args:
        img (_type_): assume img is unormalized
        cam (_type_): _description_
        labels (_type_): _description_

    Returns:
        _type_: _description_
    """

    feature_masks = threshold(cam, inverse=False) # all the features except the most important one
    feature_masks = np.expand_dims(feature_masks, axis=1)
    mask = np.broadcast_to(feature_masks, shape=(feature_masks.shape[0], 3, feature_masks.shape[-1], feature_masks.shape[-1]))

    mean = np.array([constants.IMGNET_DATA_MEAN_R, 
                     constants.IMGNET_DATA_MEAN_G, 
                     constants.IMGNET_DATA_MEAN_B]).reshape(1,3,1,1)
    perturbed_image = (1-feature_masks)*img + mean*mask
    plt.imshow(np.transpose(perturbed_image[-1,:], (1,2,0)))

    perturbed_image = torch.from_numpy(perturbed_image)
    for i in range(perturbed_image.shape[0]):
        inplace_normalize(perturbed_image[i, :])

    # plt.imshow(np.transpose(perturbed_image[-1,:], (1,2,0)))
    perturbed_image.to(dtype=constants.DTYPE, device=constants.DEVICE)
    return perturbed_image.requires_grad_(True)


def get_explanation_map(exp_map: Callable, img, cam, inplace_normalize=imgnet_inplace_transform):
    """
    Args:
        exp_map (Callable): either hard_threshold_explanation_map or soft_explanation_map
        img (tensor): _description_
        cam (tensor): _description_
    """
    # explanation_map = img*threshold(cam)
    explanation_map = exp_map(img, cam, inplace_normalize)
    return explanation_map

def segmentation_evaluation():
    pass

def model_metric_evaluation(args, val_set, val_loader, model, normalize_transform):
    '''
    log the A.D, I.C, and Confidence Drop scores

    for relevance cam
    '''
    filenames = val_set.dataset.imgs
    indices = val_set.indices
    STARTING_INDEX = 0

    axiom_style_ad_logger = Axiom_style_confidence_drop_logger()

    for x, y in val_loader:

        cams, Yci = model(x, mode=args.target_layer, target_class=[None], axiomMode=True if args.XRelevanceCAM else False)
        cams = cams[0] # cams for each image

        # NOTE: retrieve the one that correctly classified only
        # x, y = get_correct_predictions(Yci, x, y)
        cams = tensor2image(cams)
    
        print('--------- Forward Passing the Explanation Maps ------------')
        original_imgs = get_all_imgs(filenames, indices=indices[STARTING_INDEX:x.shape[0]])
        xmaps = get_explanation_map(axiom_paper_average_drop_explanation_map, original_imgs, cams, normalize_transform)
        _, Oci = model(xmaps, mode=args.target_layer, target_class=[None], axiomMode=True if args.XRelevanceCAM else False)
        Oci = torch.max(Oci, dim=1)[0].unsqueeze(1)

        # collect metrics data
        Yci = Yci.detach().numpy() if constants.WORK_ENV == 'LOCAL' else Yci.cpu().detach().numpy()
        Oci = Oci.detach().numpy() if constants.WORK_ENV == 'LOCAL' else Oci.cpu().detach().numpy()
        axiom_style_ad_logger.update(Yci, Oci)
        
        STARTING_INDEX += x.shape[0]

    return

def confidence_drop_evaluation():
    pass

def average_increase_evaluation():
    pass

def average_drop_evaluation():
    pass

### Metrics

class Axiom_style_confidence_drop_logger:
    def __init__(self):
        
        # for global statistics
        self.drop = 0
        self.N = 0

        # for local statistics
        self.per_batch_stats = 0
    def update(self, I, I_tild):
        n = I.shape[0] # batch size
        batch_drop =  np.sum((I - I_tild) / I, axis=0)
        self.per_batch_stats = 100*batch_drop / n

        self.drop += batch_drop
        self.N += n

    def get_average(self):
        return self.drop / self.N

class Increase_confidence_score:
    def __init__(self):
        self.drop = 0
        self.N = 0
        self.per_batch_stats = 0
    

class Average_drop_score:
    def __init__(self):
        self.drop = 0
        self.N = 0
        self.per_batch_stats = 0

