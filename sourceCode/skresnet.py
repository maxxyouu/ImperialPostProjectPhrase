"""
Reproduce the Selective-Kernel Resnet / Resnet Varient architecture

Adapt from: https://github.com/rwightman/pytorch-image-models/tree/master/timm/models
"""
from layers import *
from utils import build_model_with_cfg
from skresnetBlocks import SelectiveKernelBottleneck, SelectiveKernelBasic
from ResNetLocal import ResNet


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv1', 'classifier': 'fc',
        **kwargs
    }

default_cfgs = {
    'skresnet18': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/skresnet18_ra-4eec2804.pth'),
    'skresnet34': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/skresnet34_ra-bdc0ccde.pth'),
    'skresnet50': _cfg(),
    'skresnet50d': _cfg(
        first_conv='conv1.0'),
    'skresnext50_32x4d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/skresnext50_ra-f40e40bf.pth'),
}


def _create_skresnet(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(ResNet, variant, pretrained, **kwargs)

def skresnet18(pretrained=False, **kwargs):
    """Constructs a Selective Kernel ResNet-18 model.

    Different from configs in Select Kernel paper or "Compounding the Performance Improvements..." this
    variation splits the input channels to the selective convolutions to keep param count down.
    """
    sk_kwargs = dict(rd_ratio=1 / 8, rd_divisor=16, split_input=True)
    model_args = dict(
        block=SelectiveKernelBasic, layers=[2, 2, 2, 2], block_args=dict(sk_kwargs=sk_kwargs),
        zero_init_last=False, **kwargs)
    return _create_skresnet('skresnet18', pretrained, pretrained_cfg=default_cfgs['skresnet18'], **model_args)

def skresnet34(pretrained=False, **kwargs):
    """Constructs a Selective Kernel ResNet-34 model.

    Different from configs in Select Kernel paper or "Compounding the Performance Improvements..." this
    variation splits the input channels to the selective convolutions to keep param count down.
    """
    sk_kwargs = dict(rd_ratio=1 / 8, rd_divisor=16, split_input=True)
    model_args = dict(
        block=SelectiveKernelBasic, layers=[3, 4, 6, 3], block_args=dict(sk_kwargs=sk_kwargs),
        zero_init_last=False, **kwargs)
    return _create_skresnet('skresnet34', pretrained, pretrained_cfg=default_cfgs['skresnet34'], **model_args)

def skresnet50(pretrained=True, **kwargs):
    """Constructs a Select Kernel ResNet-50 model.

    Different from configs in Select Kernel paper or "Compounding the Performance Improvements..." this
    variation splits the input channels to the selective convolutions to keep param count down.
    """
    sk_kwargs = dict(split_input=True)
    model_args = dict(
        block=SelectiveKernelBottleneck, layers=[3, 4, 6, 3], block_args=dict(sk_kwargs=sk_kwargs),
        zero_init_last=False, **kwargs)
    return _create_skresnet('skresnet50', pretrained, pretrained_cfg=default_cfgs['skresnet50'], **model_args)

def skresnet50d(pretrained=False, **kwargs):
    """Constructs a Select Kernel ResNet-50-D model.

    Different from configs in Select Kernel paper or "Compounding the Performance Improvements..." this
    variation splits the input channels to the selective convolutions to keep param count down.
    """
    sk_kwargs = dict(split_input=True)
    model_args = dict(
        block=SelectiveKernelBottleneck, layers=[3, 4, 6, 3], stem_width=32, stem_type='deep', avg_down=True,
        block_args=dict(sk_kwargs=sk_kwargs), zero_init_last=False, **kwargs)
    return _create_skresnet('skresnet50d', pretrained, **model_args)

def skresnext50_32x4d(pretrained=False, **kwargs):
    """Constructs a Select Kernel ResNeXt50-32x4d model. This should be equivalent to
    the SKNet-50 model in the Select Kernel Paper
    """
    sk_kwargs = dict(rd_ratio=1/16, rd_divisor=32, split_input=False)
    model_args = dict(
        block=SelectiveKernelBottleneck, layers=[3, 4, 6, 3], cardinality=32, base_width=4,
        block_args=dict(sk_kwargs=sk_kwargs), zero_init_last=False, **kwargs)
    return _create_skresnet('skresnext50_32x4d', pretrained, pretrained_cfg=default_cfgs['skresnext50_32x4d'], **model_args)


if __name__ == '__main__':
    model = skresnext50_32x4d(pretrained=True)
    model.fc = Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load('./skresnext50_32x4d.pt', map_location=torch.device('cpu')))
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))