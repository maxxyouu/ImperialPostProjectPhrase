"""PyTorch ResNet

This started as a copy of https://github.com/pytorch/vision 'resnet.py' (BSD-3-Clause) with
additional dropout and dynamic global avg/max pool.

ResNeXt, SE-ResNeXt, SENet, and MXNet Gluon stem/downsample variants, tiered stems added by Ross Wightman

Copyright 2019, Ross Wightman

Adapt from: https://github.com/rwightman/pytorch-image-models
"""
from copy import deepcopy
import math
from pickle import FALSE
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from utilsTimm import build_model_with_cfg, checkpoint_seq
# from layers import AvgPool2dSame, create_classifier # create_attn
from layers import *
import constants

__all__ = ['ResNet', 'BasicBlock', 'Bottleneck']  # model_registry will add each entrypoint fn to this


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv1', 'classifier': 'fc',
        **kwargs
    }

def get_padding(kernel_size, stride, dilation=1):
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding

def create_aa(aa_layer, channels, stride=2, enable=True):
    if not aa_layer or not enable:
        return Identity()
    return aa_layer(stride) if issubclass(aa_layer, AvgPool2d) else aa_layer(channels=channels, stride=stride)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
            self, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64,
            reduce_first=1, dilation=1, first_dilation=None, act_layer=ReLU, norm_layer=BatchNorm2d,
            attn_layer=None, aa_layer=None, drop_block=None, drop_path=None):
        super(BasicBlock, self).__init__()

        assert cardinality == 1, 'BasicBlock only supports cardinality of 1'
        assert base_width == 64, 'BasicBlock does not support changing base width'
        first_planes = planes // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        self.conv1 = Conv2d(
            inplanes, first_planes, kernel_size=3, stride=1 if use_aa else stride, padding=first_dilation,
            dilation=first_dilation, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.drop_block = drop_block() if drop_block is not None else Identity()
        self.act1 = act_layer(inplace=True)
        self.aa = create_aa(aa_layer, channels=first_planes, stride=stride, enable=use_aa)

        self.conv2 = Conv2d(
            first_planes, outplanes, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = norm_layer(outplanes)

        # self.se = create_attn(attn_layer, outplanes)

        self.act2 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_path = drop_path

    def zero_init_last(self):
        nn.init.zeros_(self.bn2.weight)

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.drop_block(x)
        x = self.act1(x)
        x = self.aa(x)

        x = self.conv2(x)
        x = self.bn2(x)

        # if self.se is not None:
        #     x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act2(x)

        return x

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
            self, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64,
            reduce_first=1, dilation=1, first_dilation=None, act_layer=ReLU, norm_layer=BatchNorm2d,
            attn_layer=None, aa_layer=None, drop_block=None, drop_path=None):
        super(Bottleneck, self).__init__()

        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        self.conv1 = Conv2d(inplanes, first_planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = Conv2d(
            first_planes, width, kernel_size=3, stride=1 if use_aa else stride,
            padding=first_dilation, dilation=first_dilation, groups=cardinality, bias=False)
        self.bn2 = norm_layer(width)
        self.drop_block = drop_block() if drop_block is not None else Identity()
        self.act2 = act_layer(inplace=True)
        self.aa = create_aa(aa_layer, channels=width, stride=stride, enable=use_aa)

        self.conv3 = Conv2d(width, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)

        # self.se = create_attn(attn_layer, outplanes)

        self.act3 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_path = drop_path

        self.add = Add()

    def zero_init_last(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.drop_block(x)
        x = self.act2(x)
        x = self.aa(x)

        x = self.conv3(x)
        x = self.bn3(x)

        # if self.se is not None:
        #     x = self.se(x)

        # if self.drop_path is not None:
        #     x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)

        x = self.add([x, shortcut]) # x += shortcut
        x = self.act3(x)

        return x
    
    def relprop(self, R, alpha):
        out = self.act3.relprop(R, alpha)
        out, x = self.add.relprop(out, alpha) # x here is the shorcut out

        if self.downsample is not None:
            x = self.downsample.relprop(x, alpha)

        out = self.bn3.relprop(out, alpha)
        out = self.conv3.relprop(out, alpha)

        out = self.aa.relprop(out, alpha)
        out = self.act2.relprop(out, alpha)
        out = self.drop_block.relprop(out, alpha)
        out = self.bn2.relprop(out, alpha)
        out = self.conv2.relprop(out, alpha)

        out = self.act1.relprop(out, alpha)
        out = self.bn1.relprop(out, alpha)
        out = self.conv1.relprop(out, alpha)

        return out + x


def downsample_conv(
        in_channels, out_channels, kernel_size, stride=1, dilation=1, first_dilation=None, norm_layer=None):
    norm_layer = norm_layer or BatchNorm2d
    kernel_size = 1 if stride == 1 and dilation == 1 else kernel_size
    first_dilation = (first_dilation or dilation) if kernel_size > 1 else 1
    p = get_padding(kernel_size, stride, first_dilation)

    return Sequential(*[
        Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=p, dilation=first_dilation, bias=False),
        norm_layer(out_channels)
    ])

def downsample_avg(
        in_channels, out_channels, kernel_size, stride=1, dilation=1, first_dilation=None, norm_layer=None):
    norm_layer = norm_layer or BatchNorm2d
    avg_stride = stride if dilation == 1 else 1
    if stride == 1 and dilation == 1:
        pool = Identity()
    else:
        avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else AvgPool2d
        pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)

    return Sequential(*[
        pool,
        Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
        norm_layer(out_channels)
    ])

def make_blocks(
        block_fn, channels, block_repeats, inplanes, reduce_first=1, output_stride=32,
        down_kernel_size=1, avg_down=False, drop_block_rate=0., drop_path_rate=0., **kwargs):
    stages = []
    feature_info = []
    net_num_blocks = sum(block_repeats)
    net_block_idx = 0
    net_stride = 4
    dilation = prev_dilation = 1
    for stage_idx, (planes, num_blocks) in enumerate(zip(channels, block_repeats)):
        stage_name = f'layer{stage_idx + 1}'  # never liked this name, but weight compat requires it
        stride = 1 if stage_idx == 0 else 2
        if net_stride >= output_stride:
            dilation *= stride
            stride = 1
        else:
            net_stride *= stride

        downsample = None
        if stride != 1 or inplanes != planes * block_fn.expansion:
            down_kwargs = dict(
                in_channels=inplanes, out_channels=planes * block_fn.expansion, kernel_size=down_kernel_size,
                stride=stride, dilation=dilation, first_dilation=prev_dilation, norm_layer=kwargs.get('norm_layer'))
            downsample = downsample_avg(**down_kwargs) if avg_down else downsample_conv(**down_kwargs)

        block_kwargs = dict(reduce_first=reduce_first, dilation=dilation, **kwargs)
        blocks = []
        for block_idx in range(num_blocks):
            downsample = downsample if block_idx == 0 else None
            stride = stride if block_idx == 0 else 1
            blocks.append(block_fn(
                inplanes, planes, stride, downsample, first_dilation=prev_dilation,
                drop_path=None, **block_kwargs))
            prev_dilation = dilation
            inplanes = planes * block_fn.expansion
            net_block_idx += 1
        # sequential of custom blocks, each block must have a relprop function for CLRP to work
        stages.append((stage_name, Sequential(*blocks)))
        feature_info.append(dict(num_chs=inplanes, reduction=net_stride, module=stage_name))

    return stages, feature_info

class ResNet(nn.Module):
    """ResNet / ResNeXt / SE-ResNeXt / SE-Net

    This class implements all variants of ResNet, ResNeXt, SE-ResNeXt, and SENet that
      * have > 1 stride in the 3x3 conv layer of bottleneck
      * have conv-bn-act ordering

    This ResNet impl supports a number of stem and downsample options based on the v1c, v1d, v1e, and v1s
    variants included in the MXNet Gluon ResNetV1b model. The C and D variants are also discussed in the
    'Bag of Tricks' paper: https://arxiv.org/pdf/1812.01187. The B variant is equivalent to torchvision default.

    ResNet variants (the same modifications can be used in SE/ResNeXt models as well):
      * normal, b - 7x7 stem, stem_width = 64, same as torchvision ResNet, NVIDIA ResNet 'v1.5', Gluon v1b
      * c - 3 layer deep 3x3 stem, stem_width = 32 (32, 32, 64)
      * d - 3 layer deep 3x3 stem, stem_width = 32 (32, 32, 64), average pool in downsample
      * e - 3 layer deep 3x3 stem, stem_width = 64 (64, 64, 128), average pool in downsample
      * s - 3 layer deep 3x3 stem, stem_width = 64 (64, 64, 128)
      * t - 3 layer deep 3x3 stem, stem width = 32 (24, 48, 64), average pool in downsample
      * tn - 3 layer deep 3x3 stem, stem width = 32 (24, 32, 64), average pool in downsample

    ResNeXt
      * normal - 7x7 stem, stem_width = 64, standard cardinality and base widths
      * same c,d, e, s variants as ResNet can be enabled

    SE-ResNeXt
      * normal - 7x7 stem, stem_width = 64
      * same c, d, e, s variants as ResNet can be enabled

    SENet-154 - 3 layer deep 3x3 stem (same as v1c-v1s), stem_width = 64, cardinality=64,
        reduction by 2 on width of first bottleneck convolution, 3x3 downsample convs after first block

    Parameters
    ----------
    block : Block, class for the residual block. Options are BasicBlockGl, BottleneckGl.
    layers : list of int, number of layers in each block
    num_classes : int, default 1000, number of classification classes.
    in_chans : int, default 3, number of input (color) channels.
    output_stride : int, default 32, output stride of the network, 32, 16, or 8.
    global_pool : str, Global pooling type. One of 'avg', 'max', 'avgmax', 'catavgmax'
    cardinality : int, default 1, number of convolution groups for 3x3 conv in Bottleneck.
    base_width : int, default 64, factor determining bottleneck channels. `planes * base_width / 64 * cardinality`
    stem_width : int, default 64, number of channels in stem convolutions
    stem_type : str, default ''
        The type of stem:
          * '', default - a single 7x7 conv with a width of stem_width
          * 'deep' - three 3x3 convolution layers of widths stem_width, stem_width, stem_width * 2
          * 'deep_tiered' - three 3x3 conv layers of widths stem_width//4 * 3, stem_width, stem_width * 2
    block_reduce_first : int, default 1
        Reduction factor for first convolution output width of residual blocks, 1 for all archs except senets, where 2
    down_kernel_size : int, default 1, kernel size of residual block downsample path, 1x1 for most, 3x3 for senets
    avg_down : bool, default False, use average pooling for projection skip connection between stages/downsample.
    act_layer : nn.Module, activation layer
    norm_layer : nn.Module, normalization layer
    aa_layer : nn.Module, anti-aliasing layer
    drop_rate : float, default 0. Dropout probability before classifier, for training
    """

    def __init__(
            self, block, layers, num_classes=1000, in_chans=3, output_stride=32, global_pool='avg',
            cardinality=1, base_width=64, stem_width=64, stem_type='', replace_stem_pool=False, block_reduce_first=1,
            down_kernel_size=1, avg_down=False, act_layer=ReLU, norm_layer=BatchNorm2d, aa_layer=None,
            drop_rate=0.0, drop_path_rate=0., drop_block_rate=0., zero_init_last=True, block_args=None):
        super(ResNet, self).__init__()
        # miscellaneous stuff 
        block_args = block_args or dict()
        assert output_stride in (8, 16, 32)
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.grad_checkpointing = False

        # Stem
        deep_stem = 'deep' in stem_type
        inplanes = stem_width * 2 if deep_stem else 64
        if deep_stem:
            stem_chs = (stem_width, stem_width)
            if 'tiered' in stem_type:
                stem_chs = (3 * (stem_width // 4), stem_width)
            self.conv1 = Sequential(*[
                Conv2d(in_chans, stem_chs[0], 3, stride=2, padding=1, bias=False),
                norm_layer(stem_chs[0]),
                act_layer(inplace=True),
                Conv2d(stem_chs[0], stem_chs[1], 3, stride=1, padding=1, bias=False),
                norm_layer(stem_chs[1]),
                act_layer(inplace=True),
                Conv2d(stem_chs[1], inplanes, 3, stride=1, padding=1, bias=False)])
        else: #NOTE: with sknet, should be here is enough
            self.conv1 = Conv2d(in_chans, inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(inplanes)
        self.act1 = act_layer(inplace=True)
        self.feature_info = [dict(num_chs=inplanes, reduction=2, module='act1')]

        # Stem pooling. The name 'maxpool' remains for weight compatibility.
        if replace_stem_pool:
            self.maxpool = Sequential(*filter(None, [
                Conv2d(inplanes, inplanes, 3, stride=1 if aa_layer else 2, padding=1, bias=False),
                create_aa(aa_layer, channels=inplanes, stride=2) if aa_layer is not None else None,
                norm_layer(inplanes),
                act_layer(inplace=True)
            ]))
        else:
            if aa_layer is not None:
                if issubclass(aa_layer, AvgPool2d):
                    self.maxpool = aa_layer(2)
                else:
                    self.maxpool = Sequential(*[
                        MaxPool2d(kernel_size=3, stride=1, padding=1),
                        aa_layer(channels=inplanes, stride=2)])
            else: #NOTE: with sknet, should be here is enough
                self.maxpool = MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Feature Blocks
        channels = [64, 128, 256, 512]
        stage_modules, stage_feature_info = make_blocks(
            block, channels, layers, inplanes, cardinality=cardinality, base_width=base_width,
            output_stride=output_stride, reduce_first=block_reduce_first, avg_down=avg_down,
            down_kernel_size=down_kernel_size, act_layer=act_layer, norm_layer=norm_layer, aa_layer=aa_layer,
            drop_block_rate=drop_block_rate, drop_path_rate=drop_path_rate, **block_args)
        for stage in stage_modules:
            self.add_module(*stage)  # layer1, layer2, etc
        self.feature_info.extend(stage_feature_info)

        # Head (Pooling and Classifier)
        self.num_features = 512 * block.expansion
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

        self.init_weights(zero_init_last=zero_init_last)

        # self.dropout = Dropout(p=float(self.drop_rate))

    @torch.jit.ignore
    def init_weights(self, zero_init_last=True):
        for n, m in self.named_modules():
            if isinstance(m, Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        if zero_init_last:
            for m in self.modules():
                if hasattr(m, 'zero_init_last'):
                    m.zero_init_last()

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(stem=r'^conv1|bn1|maxpool', blocks=r'^layer(\d+)' if coarse else r'^layer(\d+)\.(\d+)')
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self, name_only=False):
        return 'fc' if name_only else self.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

    def forward_features(self, x, mode='output'):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq([self.layer1, self.layer2, self.layer3, self.layer4], x, flatten=True)
        else:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

        return x

    def forward_head(self, x, pre_logits: bool = False, mode='output'):
        x = self.global_pool(x)
        # if self.drop_rate:
            # x = self.dropout(x)
        return x if pre_logits else self.fc(x)

    def forward_head(self, x):
        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        act1 = self.act1(bn1)
        max_pooled_x = self.maxpool(act1)
        return max_pooled_x

    def forward_tail(self, x):  
        # classifier
        output = self.global_pool(x)
        # if self.drop_rate: # dropout is not need in resnet
            # x = self.dropout(x)
        output = output.view(output.size(0), -1) # reshape
        z = self.fc(output)
        return z

    def evaluate_axiom(self, x, y, z, args, weighted_feature_maps, correct_indices):
        per_image_z_difference = np.zeros([y.shape[0]])
        per_image_sensitivity= np.zeros([y.shape[0]])


        z = z.cpu().detach().numpy()
        y = y.cpu().detach().numpy()

        # always forward the head first with the original input
        with torch.no_grad():
            x = self.forward_head(x)
            if args.target_layer == 'layer1':
                layer_feature_maps = self.layer1(x) #(batch size, # feature maps, x, y)
            elif args.target_layer == 'layer2':
                x = self.layer1(x) #(batch size, # feature maps, x, y)
                layer_feature_maps = self.layer2(x)
            elif args.target_layer == 'layer3':
                x = self.layer1(x) #(batch size, # feature maps, x, y)
                x = self.layer2(x)
                layer_feature_maps = self.layer3(x)
            elif args.target_layer == 'layer4':
                x = self.layer1(x) #(batch size, # feature maps, x, y)
                x = self.layer2(x)
                x = self.layer3(x)
                layer_feature_maps = self.layer4(x)

            for k in range(layer_feature_maps.shape[1]):
                # create a copy of the input tensor and zero out a particular feature map
                layer_feature_maps_cp = torch.clone(layer_feature_maps)
                layer_feature_maps_cp[:, k, :] = torch.zeros_like(layer_feature_maps_cp[:, k, :])
                if args.target_layer == 'layer1':
                    output = self.layer2(layer_feature_maps_cp)
                    output = self.layer3(output)
                    output = self.layer4(output)
                    set_minus_z = self.forward_tail(output)
                elif args.target_layer == 'layer2':
                    output = self.layer3(layer_feature_maps_cp)
                    output = self.layer4(output)
                    set_minus_z = self.forward_tail(output)
                elif args.target_layer == 'layer3':
                    output = self.layer4(layer_feature_maps_cp)
                    set_minus_z = self.forward_tail(output)
                elif args.target_layer == 'layer4':
                    set_minus_z = self.forward_tail(layer_feature_maps_cp)

                set_minus_z = set_minus_z[correct_indices]
                set_minus_z = set_minus_z.cpu().detach().numpy()[range(set_minus_z.shape[0]), y]
                # set_minus_z = set_minus_z.cpu().detach().numpy()[range(set_minus_z.shape[0]), 1]

                set_minus_z -= set_minus_z/2
                feature_map_importance = torch.sum(weighted_feature_maps[:, k, :],dim=(1,2))[correct_indices]
                feature_map_importance = feature_map_importance.cpu().detach().numpy()
                per_image_z_difference += np.absolute(z-set_minus_z)
                per_image_sensitivity += np.absolute(z-set_minus_z-feature_map_importance)

            # # return the axiom stats

            per_image_normalized_conservation = np.absolute(z - torch.sum(weighted_feature_maps, dim=(1,2,3))[correct_indices].cpu().detach().numpy()) / np.absolute(z)
            per_image_normalized_sensitivity = per_image_sensitivity / per_image_z_difference
            return per_image_normalized_sensitivity, per_image_normalized_conservation


    def forward(self, x, mode='output', target_class = [None], eval_axiom=False, plusplusMode=False, lrp='CLRP', internal=False, attendCAM={}, alpha=1):
        """_summary_

        Args:
            x (_type_): _description_
            mode (str, optional): _description_. Defaults to 'output'.
            target_class (list, optional): _description_. Defaults to [None].
            lrp (str, optional): _description_. Defaults to 'CLRP'.
            internal (bool, optional): _description_. Defaults to False.
            attendCAM (dictionary, optional): key = list index, value = feature sized cam from the aux network
            alpha (int, optional): _description_. Defaults to 2.
            eval_axiom: None or string that define the layer "layer2" or None
        Returns:
            _type_: _description_
        """
        # x = self.forward_features(x)
        # z = self.forward_head(x)
        # x_origin = deepcopy(x)
        # Feature extractor
        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        act1 = self.act1(bn1)
        max_pooled_x = self.maxpool(act1)
        x = max_pooled_x
        # TODO: check the feature after maxed pooled here
        
        # def _inner_pass(out, layer):
        #     layer_outs = []
        #     for bottleneck in layer:
        #         out = bottleneck(out)
        #         layer_outs.append(out)
        #     return layer_outs

        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq([self.layer1, self.layer2, self.layer3, self.layer4], x, flatten=True)
        else:
            
            layer1 = self.layer1(x)
            
            layer2 = self.layer2(layer1)
            
            layer3 = self.layer3(layer2)
            
            layer4 = self.layer4(layer3)

        
        # classifier
        x = self.global_pool(layer4)
        # if self.drop_rate: # dropout is not need in resnet
            # x = self.dropout(x)
        x = x.view(x.size(0), -1) # reshape
        z = self.fc(x)

        # conver the ','-seperated string to a list
        mode = mode.split(',')
        if len(mode) == 1 and mode[0] == 'output':
            return [],  z

        R = self.CLRP(z, target_class) # COMPUTE THE CLRP SCORE FOR A PARTICULAR CLASS
        contrasted_logit = torch.clone(R)
        # backpropagate the classifier
        R = self.fc.relprop(R, alpha) 
        R = R.reshape_as(self.global_pool.Y) # reshape to the output size from global_pool
        # if self.drop_rate:
            # R = self.dropout.relprop(R, alpha)
        R4 = self.global_pool.relprop(R, alpha)

        def _lrp_partial_xgrad_weights(R, activations):
            """state of the art among the ones that I tried but visually it is bad
            this works!
            """
            R = R.cpu().detach().numpy() if constants.WORK_ENV == 'COLAB' else R.detach().numpy()
            activations = activations.cpu().detach().numpy() if constants.WORK_ENV == 'COLAB' else activations.detach().numpy()
            weights = R / (np.sum(activations, axis=(2, 3), keepdims=True) + 1e-7) # per channel division operation
            
            weights = np.sum(weights, axis=(2, 3), keepdims=True)
            return torch.tensor(weights, dtype=constants.DTYPE, device=constants.DEVICE)

        r_cams = []

        # LAYER 4 CAM
        if 'layer4' in mode:
            if plusplusMode:
                r_weight4 = _lrp_partial_xgrad_weights(R4, layer4)
            else:
                r_weight4 = torch.mean(R4, dim=(2, 3), keepdim=True)
            r_cam4 = layer4 * r_weight4
            if eval_axiom:
                return r_cam4, contrasted_logit
            # sum up the attention map
            r_cam4 = torch.sum(r_cam4, dim=(1), keepdim=True)
            r_cams.insert(0, r_cam4)
        if len(r_cams) == len(mode):
            return r_cams, z

        # LAYER 3 CAM
        R3 = self.layer4.relprop(R4, alpha)
        if 'layer3' in mode:
             # NOTE: propagate the LRP to the end of layer 3 and beginning of layer 4
            if plusplusMode:
                r_weight3 = _lrp_partial_xgrad_weights(R3, layer3)
            else:
                r_weight3 = torch.mean(R3, dim=(2, 3), keepdim=True)
            r_cam3 = layer3 * r_weight3
            if eval_axiom:
                return r_cam3, contrasted_logit
            r_cam3 = torch.sum(r_cam3, dim=(1), keepdim=True)
            r_cams.insert(0, r_cam3)

        if len(r_cams) == len(mode):
            return r_cams, z

        # LAYER 2 CAM
        R2 = self.layer3.relprop(R3, alpha)
        if 'layer2' in mode:
            
            if plusplusMode:
                r_weight2 = _lrp_partial_xgrad_weights(R2, layer2)
            else:
                r_weight2 = torch.mean(R2, dim=(2, 3), keepdim=True)
            r_cam2 = layer2 * r_weight2
            
            if eval_axiom:
                return r_cam2, contrasted_logit

            r_cam2 = torch.sum(r_cam2, dim=(1), keepdim=True)   
            r_cams.insert(0, r_cam2)
        if len(r_cams) == len(mode):
            return r_cams, z

        # LAYER 1 CAM
        R1 = self.layer2.relprop(R2, alpha)
        if 'layer1' in mode:
            
            if plusplusMode:
                r_weight1 = _lrp_partial_xgrad_weights(R1, layer1)
            else:
                r_weight1 = torch.mean(R1, dim=(2, 3), keepdim=True)
            r_cam1 = layer1 * r_weight1

            # return weighted feature map wihout aggregation
            if eval_axiom:
                return r_cam1, contrasted_logit

            r_cam1 = torch.sum(r_cam1, dim=(1), keepdim=True)
            r_cams.insert(0, r_cam1)

        if len(r_cams) == len(mode):
            return r_cams, z

        return r_cams, z
    
    def inner_layer_relprop(self, internal_ms,  stage,  R, alpha=1):
        """relevance cam of internal layer of each stage
        """
        r_cams = []
        internal_ms = internal_ms[::-1]
        for i, m in enumerate(reversed(stage)):
            # the current R belong to this cam
            r_weights = torch.mean(R, dim=(2, 3), keepdim=True)
            r_cam = internal_ms[i] * r_weights
            r_cam = torch.sum(r_cam, dim=(1), keepdim=True)
            r_cams.append(r_cam)

            R = m.relprop(R, alpha)
        r_cams = r_cams[::-1]
        return R, r_cams

    def CLRP(self, x, maxindex = [None]):
        # TODO: WRONG SOLUTION
        # if maxindex == [None]:
        #     maxindex = torch.argmax(x, dim=1)
        
        # # intiially R should be equal to the target logit score for each
        # if constants.WORK_ENV == 'COLAB':
        #     R = torch.ones(x.shape).cuda()
        # else:
        #     R = torch.ones(x.shape)

        # R /= -self.num_classes
        # for i in range(R.size(0)):
        #     R[i, maxindex[i]] = 1
        #     # R[i, maxindex[i]] = target_logits[i]
        # return R

        if maxindex == [None]:
            maxindex = torch.argmax(x, dim=1)
            target_logits = torch.max(x, dim=1)[0]
        
        # intiially R should be equal to the target logit score for each
        if constants.WORK_ENV == 'COLAB':
            R = torch.zeros(x.shape).cuda()
        else:
            R = torch.zeros(x.shape)

        for i in range(R.size(0)):
            mask = np.zeros(R[i,:].shape,dtype=bool) #np.ones_like(a,dtype=bool)
            mask[maxindex[i]] = True

            R[i, mask] = target_logits[i]
            R[i, ~mask] =  -target_logits[i] / self.num_classes

        return R
    
    def SGCLR(self, x, maxindex = [None]):
        softmax = nn.Softmax(dim=1)
        post_softmax = softmax(x)
        if maxindex == [None]:
            maxindex = torch.argmax(x, dim=1)
        R = torch.ones(x.shape).cuda()
        yt = torch.max(post_softmax)
        R = -yt * post_softmax
        for i in range(R.size(0)):
            R[i, maxindex[i]] = yt * (1-yt) # 1. #  # THIS CAUSE INVERSION OF THE HEAT MAP FOR SOME
        return R

def _create_resnet(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(ResNet, variant, pretrained, **kwargs)

#@register_model
def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    """
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2], **kwargs)
    return _create_resnet('resnet18', pretrained, pretrained_cfg=default_cfgs['resnet18'], **model_args)


#@register_model
def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    """
    model_args = dict(block=BasicBlock, layers=[3, 4, 6, 3], **kwargs)
    return _create_resnet('resnet34', pretrained, pretrained_cfg=default_cfgs['resnet34'], **model_args)

#@register_model
def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3],  **kwargs)
    return _create_resnet('resnet50', pretrained, pretrained_cfg=default_cfgs['resnet50'], **model_args)

#@register_model
def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], **kwargs)
    return _create_resnet('resnet101', pretrained, pretrained_cfg=default_cfgs['resnet101'], **model_args)

#@register_model
def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 8, 36, 3], **kwargs)
    return _create_resnet('resnet152', pretrained, pretrained_cfg=default_cfgs['resnet152'], **model_args)

#@register_model
def resnext50_32x4d(pretrained=False, **kwargs):
    """Constructs a ResNeXt50-32x4d model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], cardinality=32, base_width=4, **kwargs)
    return _create_resnet('resnext50_32x4d', pretrained, pretrained_cfg=default_cfgs['resnext50_32x4d'], **model_args)

#@register_model
def resnext101_32x4d(pretrained=False, **kwargs):
    """Constructs a ResNeXt-101 32x4d model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], cardinality=32, base_width=4, **kwargs)
    return _create_resnet('resnext101_32x4d', pretrained, pretrained_cfg=default_cfgs['resnext101_32x4d'], **model_args)


#@register_model
def resnext101_32x8d(pretrained=False, **kwargs):
    """Constructs a ResNeXt-101 32x8d model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], cardinality=32, base_width=8, **kwargs)
    return _create_resnet('resnext101_32x8d', pretrained, pretrained_cfg=default_cfgs['resnext101_32x8d'], **model_args)


#@register_model
def resnext101_64x4d(pretrained=False, **kwargs):
    """Constructs a ResNeXt101-64x4d model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], cardinality=64, base_width=4, **kwargs)
    return _create_resnet('resnext101_64x4d', pretrained, pretrained_cfg=default_cfgs['resnext101_64x4d'], **model_args)

default_cfgs = {
    # ResNet and Wide ResNet
    'resnet18': _cfg(url='https://download.pytorch.org/models/resnet18-5c106cde.pth'),
    'resnet18d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet18d_ra2-48a79e06.pth',
        interpolation='bicubic', first_conv='conv1.0'),
    'resnet34': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet34-43635321.pth'),
    'resnet34d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet34d_ra2-f8dcfcaf.pth',
        interpolation='bicubic', first_conv='conv1.0'),
    'resnet26': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet26-9aa10e23.pth',
        interpolation='bicubic'),
    'resnet26d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet26d-69e92c46.pth',
        interpolation='bicubic', first_conv='conv1.0'),
    'resnet26t': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-attn-weights/resnet26t_256_ra2-6f6fa748.pth',
        interpolation='bicubic', first_conv='conv1.0', input_size=(3, 256, 256), pool_size=(8, 8), crop_pct=0.94),
    'resnet50': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_a1_0-14fe96d1.pth',
        interpolation='bicubic', crop_pct=0.95),
    'resnet50d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet50d_ra2-464e36ba.pth',
        interpolation='bicubic', first_conv='conv1.0'),
    'resnet101': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet101_a1h-36d3f2aa.pth',
        interpolation='bicubic', crop_pct=0.95),
    'resnet101d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet101d_ra2-2803ffab.pth',
        interpolation='bicubic', first_conv='conv1.0', input_size=(3, 256, 256), pool_size=(8, 8),
        crop_pct=1.0, test_input_size=(3, 320, 320)),
    'resnet152': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet152_a1h-dc400468.pth',
        interpolation='bicubic', crop_pct=0.95),
    'resnet152d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet152d_ra2-5cac0439.pth',
        interpolation='bicubic', first_conv='conv1.0', input_size=(3, 256, 256), pool_size=(8, 8),
        crop_pct=1.0, test_input_size=(3, 320, 320)),
    'resnet200': _cfg(url='', interpolation='bicubic'),
    # ResNeXt
    'resnext50_32x4d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnext50_32x4d_a1h-0146ab0a.pth',
        interpolation='bicubic', crop_pct=0.95),
    'resnext50d_32x4d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnext50d_32x4d-103e99f8.pth',
        interpolation='bicubic',
        first_conv='conv1.0'),
    'resnext101_32x4d': _cfg(url=''),
    'resnext101_32x8d': _cfg(url='https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth'),
    'resnext101_64x4d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tpu-weights/resnext101_64x4d_c-0d0e0cc0.pth',
        interpolation='bicubic', crop_pct=1.0,  test_input_size=(3, 288, 288)),
    'tv_resnext50_32x4d': _cfg(url='https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth'),

}



if __name__ == '__main__':
    model = resnext50_32x4d(True)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
