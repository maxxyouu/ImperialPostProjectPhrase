"""
Building blocks for the final SKresnet and SKresnext

Adapt from: https://github.com/rwightman/pytorch-image-models/tree/master/timm/models
"""
from copy import deepcopy
import functools
import types
from typing import Optional, Tuple
import torch
from torch import nn as nn
import torch.nn.functional as F
import math

from layers import *

def make_divisible(v, divisor=8, min_value=None, round_limit=.9):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < round_limit * v:
        new_v += divisor
    return new_v

class BatchNormAct2d(BatchNorm2d):
    """BatchNorm + Activation
    #NOTE: Layer
    This module performs BatchNorm + Activation in a manner that will remain backwards
    compatible with weights trained with separate bn, act. This is why we inherit from BN
    instead of composing it as a .bn member.
    """
    def __init__(
            self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True,
            apply_act=True, act_layer=ReLU, inplace=True, drop_layer=None):
        super(BatchNormAct2d, self).__init__(
            num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
        # self.drop = drop_layer() if drop_layer is not None else Identity()
        # act_layer = get_act_layer(act_layer)  # string -> nn.Module
        if act_layer is not None and apply_act:
            act_args = dict(inplace=True) if inplace else {}
            self.act = ReLU(**act_args)
        else:
            self.act = Identity()

    def forward(self, x):
        # cut & paste of torch.BatchNorm2d.forward impl to avoid issues with torchscript and tracing
        assert(x.ndim == 4)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        x = F.batch_norm(
            x,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )
        # x = self.drop(x)
        x = self.act(x)
        return x
    
    def relprop(self, R, alpha=1):
        # take into account the activatin function operation of the module before calling the parent method
        # TODO: check if you need this
        R = self.act.relprop(R, alpha)
        return super().relprop(R, alpha)

class Conv2dSame(Conv2d):
    """ Tensorflow like 'SAME' convolution wrapper for 2D convolutions
    #NOTE: Layer
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dSame, self).__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
    def _conv2d_same(
            self, x, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, stride: Tuple[int, int] = (1, 1),
            padding: Tuple[int, int] = (0, 0), dilation: Tuple[int, int] = (1, 1), groups: int = 1):
        x = pad_same(x, weight.shape[-2:], stride, dilation)
        return Conv2d(x, weight, bias, stride, (0, 0), dilation, groups)

    def forward(self, x):
        return self._conv2d_same(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

# Calculate symmetric padding for a convolution
def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1, **_) -> int:
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding
# Can SAME padding for given args be done statically?
def is_static_pad(kernel_size: int, stride: int = 1, dilation: int = 1, **_):
    return stride == 1 and (dilation * (kernel_size - 1)) % 2 == 0

def get_padding_value(padding, kernel_size, **kwargs) -> Tuple[Tuple, bool]:
    dynamic = False
    if isinstance(padding, str):
        # for any string padding, the padding will be calculated for you, one of three ways
        padding = padding.lower()
        if padding == 'same':
            # TF compatible 'SAME' padding, has a performance and GPU memory allocation impact
            if is_static_pad(kernel_size, **kwargs):
                # static case, no extra overhead
                padding = get_padding(kernel_size, **kwargs)
            else:
                # dynamic 'SAME' padding, has runtime/GPU memory overhead
                padding = 0
                dynamic = True
        elif padding == 'valid':
            # 'VALID' padding, same as padding=0
            padding = 0
        else:
            # Default to PyTorch style 'same'-ish symmetric padding
            padding = get_padding(kernel_size, **kwargs)
    return padding, dynamic

def create_conv2d_pad(in_chs, out_chs, kernel_size, **kwargs):
    padding = kwargs.pop('padding', '')
    kwargs.setdefault('bias', False)
    padding, is_dynamic = get_padding_value(padding, kernel_size, **kwargs)
    if is_dynamic:
        return Conv2dSame(in_chs, out_chs, kernel_size, **kwargs)
    else:
        return Conv2d(in_chs, out_chs, kernel_size, padding=padding, **kwargs)

def create_conv2d(in_channels, out_channels, kernel_size, **kwargs):
    depthwise = kwargs.pop('depthwise', False)
    # for DW out_channels must be multiple of in_channels as must have out_channels % groups == 0
    groups = in_channels if depthwise else kwargs.pop('groups', 1)
    return create_conv2d_pad(in_channels, out_channels, kernel_size, groups=groups, **kwargs)

_NORM_ACT_MAP = dict(
    batchnorm=BatchNormAct2d,
    batchnorm2d=BatchNormAct2d,
)
_NORM_ACT_TYPES = {m for n, m in _NORM_ACT_MAP.items()}
_NORM_ACT_REQUIRES_ARG = {
    BatchNormAct2d}

def get_norm_act_layer(norm_layer, act_layer=None):
    # assert isinstance(norm_layer, (type, str,  types.FunctionType, functools.partial))
    # assert act_layer is None or isinstance(act_layer, (type, str, types.FunctionType, functools.partial))
    norm_act_kwargs = {}

    # unbind partial fn, so args can be rebound later
    if isinstance(norm_layer, functools.partial):
        norm_act_kwargs.update(norm_layer.keywords)
        norm_layer = norm_layer.func

    if isinstance(norm_layer, str):
        layer_name = norm_layer.replace('_', '').lower().split('-')[0]
        norm_act_layer = _NORM_ACT_MAP.get(layer_name, None)
    elif norm_layer in _NORM_ACT_TYPES:
        norm_act_layer = norm_layer
    elif isinstance(norm_layer,  types.FunctionType):
        # if function type, must be a lambda/fn that creates a norm_act layer
        norm_act_layer = norm_layer
    else:
        type_name = norm_layer.__name__.lower()
        if type_name.startswith('batchnorm'):
            norm_act_layer = BatchNormAct2d
        else:
            assert False, f"No equivalent norm_act layer for {type_name}"

    if norm_act_layer in _NORM_ACT_REQUIRES_ARG:
        # pass `act_layer` through for backwards compat where `act_layer=None` implies no activation.
        # In the future, may force use of `apply_act` with `act_layer` arg bound to relevant NormAct types
        norm_act_kwargs.setdefault('act_layer', act_layer)
    if norm_act_kwargs:
        norm_act_layer = functools.partial(norm_act_layer, **norm_act_kwargs)  # bind/rebind args
    return norm_act_layer

class ConvBnAct(nn.Module):
    """
    #NOTE Custom module
        # self.conv = Conv2d(in_channels, out_channels, kernel_size, stride=stride,
        #                       padding=sym_padding, dilation=dilation, groups=groups, bias=bias)
        # self.bn = BatchNormAct2d(out_channels, act_layer=act_layer, apply_act=apply_act)
    """
    def __init__(
            self, in_channels, out_channels, kernel_size=1, stride=1, padding='', dilation=1, groups=1,
            bias=False, apply_act=True, norm_layer=BatchNorm2d, act_layer=ReLU, drop_layer=None):
        super(ConvBnAct, self).__init__()

        # find the right padding to feed in (what has been down in the create_conv2d function)
        self.conv = create_conv2d(
            in_channels, out_channels, kernel_size, stride=stride, # BEFORE:  stride=1 if use_aa else stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias)
        
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        norm_kwargs = dict(drop_layer=drop_layer) if drop_layer is not None else {}
        self.bn = norm_act_layer(out_channels, apply_act=apply_act, **norm_kwargs)
    @property
    def in_channels(self):
        return self.conv.in_channels

    @property
    def out_channels(self):
        return self.conv.out_channels

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
    
    def relprop(self, R, alpha):
        out = self.bn.relprop(R, alpha)
        x = self.conv.relprop(out, alpha)
        return x


class SelectiveKernelAttn(nn.Module):
    def __init__(self, channels, num_paths=2, attn_channels=32, act_layer=ReLU, norm_layer=BatchNorm2d):
        """ Selective Kernel Attention Module

        Selective Kernel attention mechanism factored out into its own module.

        """
        super(SelectiveKernelAttn, self).__init__()
        self.num_paths = num_paths
        self.fc_reduce = Conv2d(channels, attn_channels, kernel_size=1, bias=False)
        self.bn = norm_layer(attn_channels)
        self.act = act_layer(inplace=True)
        self.fc_select = Conv2d(attn_channels, channels * num_paths, kernel_size=1, bias=False)
        self.softmax = Softmax(dim = 1)

        # for the relprop operations
        self.mean = Mean()
        self.sum = Sum()
        self.softmax_input_shape = None

    def forward(self, x):
        assert(x.shape[1] == self.num_paths)
        # x = x.sum(1).mean((2, 3), keepdim=True)
        x = self.sum(x, 1)
        x = self.mean(x, (2,3), keepdim=True)

        x = self.fc_reduce(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.fc_select(x)
        B, C, H, W = x.shape
        self.softmax_input_shape = x.shape # for the relprop operations
        x = x.view(B, self.num_paths, C // self.num_paths, H, W)
        x = self.softmax(x) #torch.softmax(x, dim=1)
        return x

    def relprop(self, R, alpha):
        out = self.softmax.relprop(R, alpha)
        out = out.view(self.softmax_input_shape)  # need to rearrange the shape
        out = self.fc_select.relprop(out, alpha)
        out = self.act.relprop(out, alpha)
        out = self.bn.relprop(out, alpha)
        out = self.fc_reduce.relprop(out, alpha)

        #handle the mean and summation part
        out = self.mean.relprop(out, dim=(2,3), alpha=alpha, keepdim=True)
        x = self.sum.relprop(out, dim=1, alpha=alpha)
        return x

class ConvNormActAa(nn.Module):
    def __init__(
            self, in_channels, out_channels, kernel_size=1, stride=1, padding='', dilation=1, groups=1,
            bias=False, apply_act=True, norm_layer=BatchNorm2d, act_layer=ReLU, aa_layer=None, drop_layer=None):
        super(ConvNormActAa, self).__init__()
        use_aa = aa_layer is not None

        self.conv = create_conv2d(
            in_channels, out_channels, kernel_size, stride=1 if use_aa else stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias)

        # NOTE for backwards compatibility with models that use separate norm and act layer definitions
        # norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        # NOTE for backwards (weight) compatibility, norm layer name remains `.bn`
        norm_kwargs = dict(drop_layer=drop_layer) if drop_layer is not None else {}
        # self.bn = norm_act_layer(out_channels, apply_act=apply_act, **norm_kwargs)
        self.bn = BatchNormAct2d(out_channels)
        self.aa = aa_layer(channels=out_channels) if stride == 2 and use_aa else Identity()

    @property
    def in_channels(self):
        return self.conv.in_channels

    @property
    def out_channels(self):
        return self.conv.out_channels

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.aa(x)
        return x

    def relprop(self, R, alpha):
        out = self.aa.relprop(R, alpha)
        out = self.bn.relprop(out, alpha)
        x = self.conv.relprop(out, alpha)
        return x

class SelectiveKernel(nn.Module):
    def __init__(self, in_channels, out_channels=None, kernel_size=None, stride=1, dilation=1, groups=1,
                 rd_ratio=1./16, rd_channels=None, rd_divisor=8, keep_3x3=True, split_input=True,
                 act_layer=ReLU, norm_layer=BatchNorm2d, aa_layer=None, drop_layer=None):
        super(SelectiveKernel, self).__init__()
        # miscellaneous stuff 
        def _kernel_valid(k):
            if isinstance(k, (list, tuple)):
                for ki in k:
                    return _kernel_valid(ki)
            assert k >= 3 and k % 2
        out_channels = out_channels or in_channels
        kernel_size = kernel_size or [3, 5]  # default to one 3x3 and one 5x5 branch. 5x5 -> 3x3 + dilation
        _kernel_valid(kernel_size)
        if not isinstance(kernel_size, list):
            kernel_size = [kernel_size] * 2
        if keep_3x3:
            dilation = [dilation * (k - 1) // 2 for k in kernel_size]
            kernel_size = [3] * len(kernel_size)
        else:
            dilation = [dilation] * len(kernel_size)
        self.num_paths = len(kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.split_input = split_input
        if self.split_input:
            assert in_channels % self.num_paths == 0
            in_channels = in_channels // self.num_paths
        groups = min(out_channels, groups)

        conv_kwargs = dict(
            stride=stride, groups=groups, act_layer=act_layer, norm_layer=norm_layer)
        # ConvBnAct  = ConvNormAct and SelectiveKernel DOES NOT use ConvNormActAa

        # NOTE: the conv ops in self.path is performed in parallel instead of sequential
        self.paths = nn.ModuleList([
            ConvBnAct(in_channels, out_channels, kernel_size=k, dilation=d, **conv_kwargs)
            for k, d in zip(kernel_size, dilation)])

        attn_channels = rd_channels or make_divisible(out_channels * rd_ratio, divisor=rd_divisor)
        self.attn = SelectiveKernelAttn(out_channels, self.num_paths, attn_channels)

        # for relProp operations
        self.sum = Sum()
        self.mean = Mean()
        self.mul = Multiply()

    def forward(self, x):
        if self.split_input:
            x_split = torch.split(x, self.in_channels // self.num_paths, 1) # dim = 1 is the channel dimention
            x_paths = [op(x_split[i]) for i, op in enumerate(self.paths)] # perform operation for each chunk of tensor
        else:
            x_paths = [op(x) for op in self.paths]
        # [batch_size, paths, channels, width, height]
        x = torch.stack(x_paths, dim=1) # stack all the operated input, that's why you see the dim 2 in the tensor
        x_attn = self.attn(x)
        x = self.mul([x, x_attn]) #x * x_attn
        x = self.sum(x, dim=1) #torch.sum(x, dim=1)
        return x
    

    def relprop(self, R, alpha):
        #TODO: something when work with batch size
        # def _unstack(_input):
        #     return torch.cat(torch.unbind((_input).squeeze(0), dim=0)).unsqueeze(0)
        def _unstack(_input):
            return torch.cat(torch.unbind(out, dim=1), dim=1)
        # NOTE: refer to the selective kernel diagram to do the LRP propagation
        out = self.sum.relprop(R, dim=1) # dim=2 as the number of paths
        out, attn = self.mul.relprop(out)
        attn_out = self.attn.relprop(attn, alpha)
        out = _unstack(attn_out + out) # [batch-size, paths, channels, width, height]


        # Handle the stack /splits/paths, the ones in nn.modulelist
        if self.split_input:
            # split the out into the corresponding chunks of size self.in_channels // self.num_paths as input
            chunk_input_size = self.in_channels
            out_split = torch.split(out, chunk_input_size, 1)

            # operation is acted on the corresponding chunk
            out_paths = [op.relprop(out_split[i], alpha) for i, op in enumerate(self.paths)]

            # stack the split back together
            x = torch.cat(out_paths, dim=1)
        else:
            # TODO: check if this makes sense
            # print('no split happens')
            chunk_input_size = self.in_channels
            out_split = torch.split(out, chunk_input_size, 1)

            # operation is acted on the corresponding chunk
            out_paths = [op.relprop(out_split[i], alpha) for i, op in enumerate(self.paths)]
            x = torch.stack(out_paths, dim=1)
            # x = torch.mean(x, dim=1)
            x = torch.sum(x, dim=1)
        return x

class SelectiveKernelBasic(nn.Module):
    expansion = 1

    def __init__(
            self, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64,
            sk_kwargs=None, reduce_first=1, dilation=1, first_dilation=None, act_layer=ReLU,
            norm_layer=BatchNorm2d, attn_layer=None, aa_layer=None, drop_block=None, drop_path=None):
         # miscellaneous stuff 
        super(SelectiveKernelBasic, self).__init__()
        sk_kwargs = sk_kwargs or {}
        conv_kwargs = dict(act_layer=act_layer, norm_layer=norm_layer)
        assert cardinality == 1, 'BasicBlock only supports cardinality of 1'
        assert base_width == 64, 'BasicBlock doest not support changing base width'
        first_planes = planes // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation

        self.conv1 = SelectiveKernel(
            inplanes, first_planes, stride=stride, dilation=first_dilation,
            aa_layer=aa_layer, drop_layer=drop_block, **conv_kwargs, **sk_kwargs)
        self.conv2 = ConvBnAct(
            first_planes, outplanes, kernel_size=3, dilation=dilation, apply_act=False, **conv_kwargs)
        # self.se = create_attn(attn_layer, outplanes)
        self.act = act_layer(inplace=True)
        self.downsample = downsample
        self.drop_path = drop_path

        if downsample:
            print('has downsample')
        if drop_path:
            print('has droppath')
        self.add = Add()
    
    def zero_init_last(self):
        nn.init.zeros_(self.conv2.bn.weight)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        # if self.se is not None:
        #     x = self.se(x)
        # if self.drop_path is not None:
        #     x = self.drop_path(x)
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x = self.add([x, shortcut]) # x += shortcut
        x = self.act(x)
        return x

    def relprop(self, R, alpha):
        out = self.act.relprop(R, alpha)
        out, x = self.add.relprop(out, alpha)
        if self.downsample is not None:
            x = self.downsample.relprop(x, alpha)
        # if self.se is not None:
        #     x = self.se.relprop(x, alpha)
        out = self.conv2.relprop(out, alpha)
        x1 = self.conv1.relprop(out, alpha)
        return x1 + x

class SelectiveKernelBottleneck(nn.Module):
    expansion = 4

    def __init__(
            self, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64, sk_kwargs=None,
            reduce_first=1, dilation=1, first_dilation=None, act_layer=ReLU, norm_layer=BatchNorm2d,
            attn_layer=None, aa_layer=None, drop_block=None, drop_path=None):

        # miscellaneous stuff 
        super(SelectiveKernelBottleneck, self).__init__()
        sk_kwargs = sk_kwargs or {}
        conv_kwargs = dict(act_layer=act_layer, norm_layer=norm_layer)
        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation

        # equilvalent of ConvNormAct
        self.conv1 = ConvBnAct(inplanes, first_planes, kernel_size=1, **conv_kwargs)
        self.conv2 = SelectiveKernel(
                    first_planes, width, stride=stride, dilation=first_dilation, groups=cardinality,
                    aa_layer=aa_layer, drop_layer=drop_block, **conv_kwargs, **sk_kwargs)
        self.conv3 = ConvBnAct(width, outplanes, kernel_size=1, apply_act=False, **conv_kwargs)
        # self.se = create_attn(attn_layer, outplanes) # used for SE module but None by default
        self.act = act_layer(inplace=True) # activation function
        self.downsample = downsample
        self.drop_path = drop_path
        self.add = Add()

    def zero_init_last(self):
        nn.init.zeros_(self.conv3.bn.weight)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # if self.se is not None:
        #     x = self.se(x)
        # if self.drop_path is not None:
        #     x = self.drop_path(x)
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x = self.add([x, shortcut]) # x += shortcut
        x = self.act(x)
        return x

    def relprop(self, R, alpha):
        out = self.act.relprop(R, alpha)
        out, x = self.add.relprop(out, alpha) # x here is the shorcut out
        if self.downsample is not None:
            x = self.downsample.relprop(x, alpha)
        # if self.se is not None:
        #     x = self.se.relprop(x, alpha)
        out = self.conv3.relprop(out, alpha)
        out = self.conv2.relprop(out, alpha)
        x1 = self.conv1.relprop(out, alpha)

        return x1 + x
