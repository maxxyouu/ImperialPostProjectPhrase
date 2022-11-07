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
    return torch.from_numpy(img*np.expand_dims(threshold(cam), axis=1))

def hard_inverse_threshold_explanation_map(img, cam):
    """
    used for select which layer of the relevance cam to be used
    """
    return torch.from_numpy(img*np.expand_dims(threshold(cam, inverse=True), axis=1))

def soft_explanation_map(img, cam):
    """in the grad cam paper
    used for examine the metrics AD, AI
    """
    return torch.from_numpy(img * np.expand_dims(np.maximum(cam, 0), axis=1))

def axiom_paper_average_drop_explanation_map(img, cam):
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
    # plt.imshow(np.transpose(perturbed_image[-1,:], (1,2,0)))

    perturbed_image = torch.from_numpy(perturbed_image)

    # plt.imshow(np.transpose(perturbed_image[-1,:], (1,2,0)))
    return perturbed_image


def get_explanation_map(exp_map: Callable, img, cam, inplace_normalize):
    """
    Args:
        exp_map (Callable): either hard_threshold_explanation_map or soft_explanation_map
        img (tensor): _description_
        cam (tensor): _description_
    """
    # explanation_map = img*threshold(cam)
    perturbed_image = exp_map(img, cam)
    for i in range(perturbed_image.shape[0]):
        inplace_normalize(perturbed_image[i, :])
    perturbed_image = perturbed_image.to(dtype=constants.DTYPE, device=constants.DEVICE)
    return perturbed_image.requires_grad_(True)

def segmentation_evaluation():
    pass

def model_metric_evaluation(args, val_set, val_loader, model, normalize_transform, metrics_logger:Callable, xmap_extractor:Callable):
    '''
    log the A.D, I.C, and Confidence Drop scores

    for relevance cam
    '''
    filenames = val_set.dataset.imgs
    indices = val_set.indices
    STARTING_INDEX = 0

    for x, y in val_loader:
        x = x.to(device=constants.DEVICE, dtype=constants.DTYPE)  # move to device, e.g. GPU
        y = y.to(device=constants.DEVICE, dtype=constants.DTYPE)
        cams, Yci = model(x, mode=args.target_layer, target_class=[None], axiomMode=True if args.XRelevanceCAM else False)
        cams = cams[0] # cams for each image

        # NOTE: retrieve the one that correctly classified only
        if args.correctPredictionsOnly:
            x, Yci, cams = get_correct_predictions(Yci, x, y, cams)
        else:
            # only need to highest Yci scores
            Yci = torch.max(Yci, dim=1)[0].unsqueeze(1)

        cams = tensor2image(cams)
    
        print('--------- Forward Passing the Explanation Maps ------------')
        original_imgs = get_all_imgs(filenames, indices=indices[STARTING_INDEX:STARTING_INDEX+x.shape[0]])
        xmaps = get_explanation_map(xmap_extractor, original_imgs, cams, normalize_transform)
        _, Oci = model(xmaps, mode=args.target_layer, target_class=[None], axiomMode=True if args.XRelevanceCAM else False)
        Oci = torch.max(Oci, dim=1)[0].unsqueeze(1)
        # collect metrics data
        Yci = Yci.cpu().detach().numpy()
        Oci = Oci.cpu().detach().numpy()
        metrics_logger.update(Yci, Oci)   

        # print per batch statistic
        print('Per batch statistics: {}'.format(metrics_logger.per_batch_stats.item()))
        STARTING_INDEX += x.shape[0]

    # print the stats:
    print('Average Statistics: {}'.format(metrics_logger.get_average()))
    return


### Metrics

class Axiom_style_confidence_drop_logger:
    def __init__(self):
        
        # for global statistics
        self.drop = 0
        self.N = 0

        # for local statistics
        self.per_batch_stats = 0
        print('Axiom-style average confidence drop metrics')

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
        print('Increase in Confidence metric')


    def get_avg(self):
        return 100 * self.drop / self.N
    
    def update(self, Yci, Oci):
        """
        CASE: the explanation map remove the confounded features that confuse the classifier.
        => Oci > Yci
        """
        indicator = Yci < Oci
        batch_size = indicator.shape[0]
        
        # aggregate the batch statistics    
        increase_in_confidence = np.sum(indicator, axis=0)
        self.per_batch_stats = 100 * increase_in_confidence / batch_size
        
        self.N += batch_size
        self.drop += increase_in_confidence
    
    
class Average_drop_score:
    def __init__(self):
        self.drop = 0
        self.N = 0
        self.per_batch_stats = 0
        print('Average drop metric')

    
    def get_avg(self):
        return 100 * self.drop / self.N

    def update(self, Yci, Oci):
        """case where Yci > Oci: 
        
        -Oci is the score with explanation map only and should maintain high confidence score
        because it includes the most relevance part from the full input image

        -A.D the lower the better
        """
        percentage_drop = (Yci - Oci) / Yci
        percentage_drop = np.maximum(percentage_drop, 0)

        # aggregate the batch statistics
        batch_size = percentage_drop.shape[0]
        batch_pd = np.sum(percentage_drop, axis=0)
        self.per_batch_stats = 100*batch_pd / batch_size
    
        self.N += batch_size
        self.drop += batch_pd