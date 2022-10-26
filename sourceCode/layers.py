"""
Customed Layer to produce the RelavenceCAM
Reference:
Adapt from: https://github.com/mongeoroo/Relevance-CAM
"""
from typing import List
import torch
from torch import nn as nn
import torch.nn.functional as F
from itertools import repeat
import collections.abc
import math

def safe_divide(a, b):
    return a / (b + b.eq(0).type(b.type()) * 1e-9) * b.ne(0).type(b.type())

class Sequential(nn.Sequential):
    def relprop(self, R, alpha = 1):
        for m in reversed(self._modules.values()):
            R = m.relprop(R, alpha)
        return R
    def RAP_relprop(self, Rp):
        for m in reversed(self._modules.values()):
            Rp = m.RAP_relprop(Rp)
        return Rp

def forward_hook(self, input, output):
    if type(input[0]) in (list, tuple): # handle the case where in the input is a list like in Add below
        self.X = []  # for each module that register a hook, it first clear it then append
        for i in input[0]: # each structure in the input[0]
            x = i.detach()
            x.requires_grad = True
            self.X.append(x)
    else:
        self.X = input[0].detach()
        self.X.requires_grad = True

    self.Y = output


class RelProp(nn.Module):
    def __init__(self):
        super(RelProp, self).__init__()
        # if not self.training:
        self.register_forward_hook(forward_hook)

    def gradprop(self, Z, X, S):
        C = torch.autograd.grad(Z, X, S, retain_graph=True)
        return C

    def relprop(self, R, alpha = 1):
        return R
    def m_relprop(self, R,pred,  alpha = 1):
        return R
    def RAP_relprop(self, R_p):
        return R_p

class RelPropSimple(RelProp):
    def relprop(self, R, alpha = 1):
        Z = self.forward(self.X)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)

        if torch.is_tensor(self.X) == False:
            outputs = []
            outputs.append(self.X[0] * C[0])
            outputs.append(self.X[1] * C[1])
        else:
            outputs = self.X * C[0]
        return outputs

    def RAP_relprop(self, R_p):
        def backward(R_p):
            Z = self.forward(self.X)
            Sp = safe_divide(R_p, Z)

            Cp = self.gradprop(Z, self.X, Sp)[0]
            if torch.is_tensor(self.X) == False:
                Rp = []
                Rp.append(self.X[0] * Cp)
                Rp.append(self.X[1] * Cp)
            else:
                Rp = self.X * (Cp)
            return Rp
        if torch.is_tensor(R_p) == False:
            idx = len(R_p)
            tmp_R_p = R_p
            Rp = []
            for i in range(idx):
                Rp_tmp = backward(tmp_R_p[i])
                Rp.append(Rp_tmp)
        else:
            Rp = backward(R_p)
        return Rp

class Clone(RelProp):
    def forward(self, input, num):
        self.__setattr__('num', num)
        outputs = []
        for _ in range(num):
            outputs.append(input)

        return outputs

    def relprop(self, R, alpha = 1):
        Z = []
        for _ in range(self.num):
            Z.append(self.X)
        S = [safe_divide(r, z) for r, z in zip(R, Z)]
        C = self.gradprop(Z, self.X, S)[0]

        R = self.X * C

        return R

    def RAP_relprop(self, R_p):
        def backward(R_p):
            Z = []
            for _ in range(self.num):
                Z.append(self.X)

            Spp = []
            Spn = []

            for z, rp, rn in zip(Z, R_p):
                Spp.append(safe_divide(torch.clamp(rp, min=0), z))
                Spn.append(safe_divide(torch.clamp(rp, max=0), z))

            Cpp = self.gradprop(Z, self.X, Spp)[0]
            Cpn = self.gradprop(Z, self.X, Spn)[0]

            Rp = self.X * (Cpp * Cpn)

            return Rp
        if torch.is_tensor(R_p) == False:
            idx = len(R_p)
            tmp_R_p = R_p
            Rp = []
            for i in range(idx):
                Rp_tmp = backward(tmp_R_p[i])
                Rp.append(Rp_tmp)
        else:
            Rp = backward(R_p)
        return Rp


class Identity(nn.Identity, RelProp):
    pass

class ReLU(nn.ReLU, RelProp):
    pass

class Dropout(nn.Dropout, RelProp):
    pass

class MaxPool2d(nn.MaxPool2d, RelPropSimple):
    pass

class AvgPool2d(nn.AvgPool2d, RelPropSimple):
    pass

class Softmax(nn.Softmax, RelPropSimple):
    pass

class Multiply(RelPropSimple):
    def forward(self, inputs):
        return torch.mul(*inputs)

    def relprop(self, R, alpha = 1):
        x0 = torch.clamp(self.X[0],min=0)
        x1 = torch.clamp(self.X[1],min=0)
        x = [x0,x1]
        Z = self.forward(x)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, x, S)

        outputs = []
        outputs.append(x[0] * C[0])
        outputs.append(x[1] * C[1])

        return outputs

    # def relprop(self, R, alpha = 1):
    #     Z = self.forward(self.X)
    #     S = safe_divide(R, Z)
    #     C = self.gradprop(Z, self.X, S)

    #     if torch.is_tensor(self.X) == False:
    #         outputs = []
    #         outputs.append(self.X[0] * C[0])
    #         outputs.append(self.X[1] * C[1])
    #     else:
    #         outputs = self.X * C[0]
    #     return outputs

class Add(RelPropSimple):
    def forward(self, inputs):
        # *input will delist the list of tensors to be added
        return torch.add(*inputs)

class Sum(RelPropSimple):
    def forward(self, inputs, dim=1, keepdim=False):
        return torch.sum(inputs, dim, keepdim=keepdim)
    # def relprop(self, R, alpha = 1):
    #     x0 = torch.clamp(self.X[0],min=0)
    #     x1 = torch.clamp(self.X[1],min=0)
    #     x = [x0,x1]
    #     Z = self.forward(x)
    #     S = safe_divide(R, Z)
    #     C = self.gradprop(Z, x, S)

    #     outputs = []
    #     outputs.append(x[0] * C[0])
    #     outputs.append(x[1] * C[1])

    #     return outputs
    def relprop(self, R, dim, alpha = 1, keepdim=False):
        Z = self.forward(self.X, dim, keepdim=keepdim)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)

        if torch.is_tensor(self.X) == False:
            outputs = []
            outputs.append(self.X[0] * C[0])
            outputs.append(self.X[1] * C[1])
        else:
            outputs = self.X * C[0]
        return outputs

class Mean(RelPropSimple):
    def forward(self, inputs, dim, keepdim=False):
        return torch.mean(inputs, dim, keepdim=keepdim)

    def relprop(self, R, dim, alpha = 1, keepdim=False):
        Z = self.forward(self.X, dim, keepdim=keepdim)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)

        if torch.is_tensor(self.X) == False:
            outputs = []
            outputs.append(self.X[0] * C[0])
            outputs.append(self.X[1] * C[1])
        else:
            outputs = self.X * C[0]
        return outputs


class SelectAdaptivePool2d(nn.Module):
    """Selectable global pooling layer with dynamic input kernel size
    """
    def __init__(self, output_size=1, pool_type='fast', flatten=False):
        super(SelectAdaptivePool2d, self).__init__()
        self.pool_type = pool_type or ''  # convert other falsy values to empty string for consistent TS typing
        self.flatten = nn.Flatten(1) if flatten else nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d(output_size)

    def is_identity(self):
        return not self.pool_type

    def forward(self, x):
        x = self.pool(x)
        x = self.flatten(x)
        return x

    def feat_mult(self):
        return adaptive_pool_feat_mult(self.pool_type)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + 'pool_type=' + self.pool_type \
               + ', flatten=' + str(self.flatten) + ')'

# Calculate asymmetric TensorFlow-like 'SAME' padding for a convolution
def get_same_padding(x: int, k: int, s: int, d: int):
    return max((math.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)

# Dynamically pad input x with 'SAME' padding for conv with specified args
def pad_same(x, k: List[int], s: List[int], d: List[int] = (1, 1), value: float = 0):
    ih, iw = x.size()[-2:]
    pad_h, pad_w = get_same_padding(ih, k[0], s[0], d[0]), get_same_padding(iw, k[1], s[1], d[1])
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2], value=value)
    return x

# TODO: pending to add RELPROP
class AvgPool2dSame(nn.AvgPool2d):
    """ Tensorflow like 'SAME' wrapper for 2D average pooling
    """
    def __init__(self, kernel_size: int, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
        # From PyTorch internals
        def _ntuple(n):
            def parse(x):
                if isinstance(x, collections.abc.Iterable):
                    return x
                return tuple(repeat(x, n))
            return parse

        to_2tuple = _ntuple(2)
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        super(AvgPool2dSame, self).__init__(kernel_size, stride, (0, 0), ceil_mode, count_include_pad)

    def forward(self, x):
        x = pad_same(x, self.kernel_size, self.stride)
        return F.avg_pool2d(
            x, self.kernel_size, self.stride, self.padding, self.ceil_mode, self.count_include_pad)

def adaptive_pool_feat_mult(pool_type='avg'):
    if pool_type == 'catavgmax':
        return 2
    else:
        return 1

def _create_pool(num_features, num_classes, pool_type='avg', use_conv=False):
    flatten_in_pool = not use_conv  # flatten when we use a Linear layer after pooling
    if not pool_type:
        assert num_classes == 0 or use_conv,\
            'Pooling can only be disabled if classifier is also removed or conv classifier is used'
        flatten_in_pool = False  # disable flattening if pooling is pass-through (no pooling)
    global_pool = SelectAdaptivePool2d(pool_type=pool_type, flatten=flatten_in_pool)
    num_pooled_features = num_features * global_pool.feat_mult()
    return global_pool, num_pooled_features


def _create_fc(num_features, num_classes, use_conv=False):
    if num_classes <= 0:
        fc = Identity()  # pass-through (no classifier)
    elif use_conv:
        fc = Conv2d(num_features, num_classes, 1, bias=True)
    else:
        fc = Linear(num_features, num_classes, bias=True)
    return fc


def create_classifier(num_features, num_classes, pool_type='avg', use_conv=False):
    # global_pool, num_pooled_features = _create_pool(num_features, num_classes, pool_type, use_conv=use_conv)
    # fc = _create_fc(num_pooled_features, num_classes, use_conv=use_conv)

    # the following are the only mechanisms we need
    global_pool = AdaptiveAvgPool2d((1, 1))
    fc = Linear(num_features, num_classes, bias=True)
    return global_pool, fc

class ConvTranspose2d(nn.ConvTranspose2d, RelProp):
    def relprop(self, R, alpha = 1):

        pw = torch.clamp(self.weight, min=0)
        nw = torch.clamp(self.weight, max=0)
        px = torch.clamp(self.X, min=0)
        nx = torch.clamp(self.X, max=0)

        def f(w1, x1):
            Z1 = F.conv_transpose2d(x1, w1, bias=None, stride=self.stride, padding=self.padding, output_padding=self.output_padding)
            S1 = safe_divide(R, Z1)
            C1 = x1 * self.gradprop(Z1, x1, S1)[0]
            return C1

        activator_relevances = f(pw, px)
        R = activator_relevances
        return R

class AdaptiveAvgPool2d(nn.AdaptiveAvgPool2d, RelProp):
    def relprop(self, R, alpha=1):
        px = torch.clamp(self.X, min=0)

        def f(x1):
            Z1 = F.adaptive_avg_pool2d(x1, self.output_size)

            S1 = safe_divide(R, Z1)

            C1 = x1 * self.gradprop(Z1, x1, S1)[0]

            return C1

        activator_relevances = f(px)
        out = activator_relevances

        return out


class BatchNorm2d(nn.BatchNorm2d, RelProp):
    def relprop(self, R, alpha = 1):
        X = self.X  # self.X comes from the forward_hook function 
        beta = 1 - alpha
        weight = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3) / (
            (self.running_var.unsqueeze(0).unsqueeze(2).unsqueeze(3).pow(2) + self.eps).pow(0.5))
        Z = X * weight + 1e-9
        S = R / Z
        Ca = S * weight
        R = self.X * (Ca)
        return R

    def RAP_relprop(self, R_p):
        def f(R, w1, x1):
            Z1 = x1 * w1
            S1 = safe_divide(R, Z1) * w1
            C1 = x1 * S1
            return C1
        def backward(R_p):
            X = self.X

            weight = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3) / (
                (self.running_var.unsqueeze(0).unsqueeze(2).unsqueeze(3).pow(2) + self.eps).pow(0.5))

            if torch.is_tensor(self.bias):
                bias = self.bias.unsqueeze(-1).unsqueeze(-1)
                bias_p = safe_divide(bias * R_p.ne(0).type(self.bias.type()),
                                     R_p.ne(0).type(self.bias.type()).sum(dim=[2, 3], keepdim=True))
                R_p = R_p - bias_p

            Rp = f(R_p, weight, X)

            if torch.is_tensor(self.bias):
                Bp = f(bias_p, weight, X)

                Rp = Rp + Bp

            return Rp

        if torch.is_tensor(R_p) == False:
            idx = len(R_p)
            tmp_R_p = R_p
            Rp = []
            for i in range(idx):
                Rp_tmp = backward(tmp_R_p[i])
                Rp.append(Rp_tmp)
        else:
            Rp = backward(R_p)
        return Rp

class Linear(nn.Linear, RelProp):
    def relprop(self, R, alpha = 1):
        beta = alpha - 1
        pw = torch.clamp(self.weight, min=0)
        nw = torch.clamp(self.weight, max=0)
        px = torch.clamp(self.X, min=0)
        nx = torch.clamp(self.X, max=0)

        # def f(w1, w2, x1, x2):
        #     Z1 = F.linear(x1, w1)
        #     Z2 = F.linear(x2, w2)
        #     S1 = safe_divide(R, Z1)
        #     S2 = safe_divide(R, Z2)
        #     C1 = x1 * self.gradprop(Z1, x1, S1)[0]
        #     C2 = x2 * self.gradprop(Z2, x2, S2)[0]
        #     return C1 #+ C2

        def f(w1, w2, x1, x2):
            Z1 = F.linear(x1, w1)
            Z2 = F.linear(x2, w2)
            Z = Z1 + Z2
            S = safe_divide(R, Z)
            C1 = x1 * self.gradprop(Z1, x1, S)[0]
            C2 = x2 * self.gradprop(Z2, x2, S)[0]
            return C1 + C2


        activator_relevances = f(pw, nw, px, nx)
        inhibitor_relevances = f(nw, pw, px, nx)

        out = alpha * activator_relevances - beta*inhibitor_relevances

        return out

    def RAP_relprop(self, R_p):
        def shift_rel(R, R_val):
            R_nonzero = torch.ne(R, 0).type(R.type())
            shift = safe_divide(R_val, torch.sum(R_nonzero, dim=-1, keepdim=True)) * torch.ne(R, 0).type(R.type())
            K = R - shift
            return K
        def pos_prop(R, Za1, Za2, x1):
            R_pos = torch.clamp(R, min=0)
            R_neg = torch.clamp(R, max=0)
            S1 = safe_divide((R_pos * safe_divide((Za1 + Za2), Za1 + Za2)), Za1)
            C1 = x1 * self.gradprop(Za1, x1, S1)[0]
            S1n = safe_divide((R_neg * safe_divide((Za1 + Za2), Za1 + Za2)), Za1)
            C1n = x1 * self.gradprop(Za1, x1, S1n)[0]
            S2 = safe_divide((R_pos * safe_divide((Za2), Za1 + Za2)), Za2)
            C2 = x1 * self.gradprop(Za2, x1, S2)[0]
            S2n = safe_divide((R_neg * safe_divide((Za2), Za1 + Za2)), Za2)
            C2n = x1 * self.gradprop(Za2, x1, S2n)[0]
            Cp = C1 + C2
            Cn = C2n + C1n

            C = (Cp + Cn)
            C = shift_rel(C, C.sum(dim=-1,keepdim=True)-R.sum(dim=-1,keepdim=True))
            return C
        def f(R, w1, w2, x1, x2):
            R_nonzero = R.ne(0).type(R.type())
            Za1 = F.linear(x1, w1) * R_nonzero
            Za2 = - F.linear(x1, w2) * R_nonzero

            Zb1 = - F.linear(x2, w1) * R_nonzero
            Zb2 = F.linear(x2, w2) * R_nonzero

            C1 = pos_prop(R, Za1, Za2, x1)
            C2 = pos_prop(R, Zb1, Zb2, x2)

            return C1 + C2

        def first_prop(pd, px, nx, pw, nw):
            Rpp = F.linear(px, pw) * pd
            Rpn = F.linear(px, nw) * pd
            Rnp = F.linear(nx, pw) * pd
            Rnn = F.linear(nx, nw) * pd
            Pos = (Rpp + Rnn).sum(dim=-1, keepdim=True)
            Neg = (Rpn + Rnp).sum(dim=-1, keepdim=True)

            Z1 = F.linear(px, pw)
            Z2 = F.linear(px, nw)
            Z3 = F.linear(nx, pw)
            Z4 = F.linear(nx, nw)

            S1 = safe_divide(Rpp, Z1)
            S2 = safe_divide(Rpn, Z2)
            S3 = safe_divide(Rnp, Z3)
            S4 = safe_divide(Rnn, Z4)
            C1 = px * self.gradprop(Z1, px, S1)[0]
            C2 = px * self.gradprop(Z2, px, S2)[0]
            C3 = nx * self.gradprop(Z3, nx, S3)[0]
            C4 = nx * self.gradprop(Z4, nx, S4)[0]
            bp = self.bias * pd * safe_divide(Pos, Pos + Neg)
            bn = self.bias * pd * safe_divide(Neg, Pos + Neg)
            Sb1 = safe_divide(bp, Z1)
            Sb2 = safe_divide(bn, Z2)
            Cb1 = px * self.gradprop(Z1, px, Sb1)[0]
            Cb2 = px * self.gradprop(Z2, px, Sb2)[0]
            return C1 + C4 + Cb1 + C2 + C3 + Cb2
        def backward(R_p, px, nx, pw, nw):
            # dealing bias
            # if torch.is_tensor(self.bias):
            #     bias_p = self.bias * R_p.ne(0).type(self.bias.type())
            #     R_p = R_p - bias_p

            Rp = f(R_p, pw, nw, px, nx)

            # if torch.is_tensor(self.bias):
            #     Bp = f(bias_p, pw, nw, px, nx)
            #
            #     Rp = Rp + Bp
            return Rp
        def redistribute(Rp_tmp):
            Rp = torch.clamp(Rp_tmp, min=0)
            Rn = torch.clamp(Rp_tmp, max=0)
            R_tot = (Rp - Rn).sum(dim=-1, keepdim=True)
            Rp_tmp3 = safe_divide(Rp, R_tot) * (Rp + Rn).sum(dim=-1, keepdim=True)
            Rn_tmp3 = -safe_divide(Rn, R_tot) * (Rp + Rn).sum(dim=-1, keepdim=True)
            return Rp_tmp3 + Rn_tmp3
        pw = torch.clamp(self.weight, min=0)
        nw = torch.clamp(self.weight, max=0)
        X = self.X
        px = torch.clamp(X, min=0)
        nx = torch.clamp(X, max=0)
        if torch.is_tensor(R_p) == True and R_p.max() == 1:  ## first propagation
            pd = R_p

            Rp_tmp = first_prop(pd, px, nx, pw, nw)
            A =  redistribute(Rp_tmp)

            return A
        else:
            Rp = backward(R_p, px, nx, pw, nw)


        return Rp
class Conv2d(nn.Conv2d, RelProp):
    def gradprop2(self, DY, weight):
        Z = self.forward(self.X)

        output_padding = self.X.size()[2] - (
                (Z.size()[2] - 1) * self.stride[0] - 2 * self.padding[0] + self.kernel_size[0])

        return F.conv_transpose2d(DY, weight, stride=self.stride, padding=self.padding, output_padding=output_padding)

    def relprop(self, R, alpha = 1):
        if self.X.shape[1] == 3:
            pw = torch.clamp(self.weight, min=0)
            nw = torch.clamp(self.weight, max=0)
            X = self.X
            L = self.X * 0 + \
                torch.min(torch.min(torch.min(self.X, dim=1, keepdim=True)[0], dim=2, keepdim=True)[0], dim=3,
                          keepdim=True)[0]
            H = self.X * 0 + \
                torch.max(torch.max(torch.max(self.X, dim=1, keepdim=True)[0], dim=2, keepdim=True)[0], dim=3,
                          keepdim=True)[0]
            Za = torch.conv2d(X, self.weight, bias=None, stride=self.stride, padding=self.padding) - \
                 torch.conv2d(L, pw, bias=None, stride=self.stride, padding=self.padding) - \
                 torch.conv2d(H, nw, bias=None, stride=self.stride, padding=self.padding) + 1e-9

            S = R / Za
            C = X * self.gradprop2(S, self.weight) - L * self.gradprop2(S, pw) - H * self.gradprop2(S, nw)
            R = C
        else:
            beta = alpha - 1
            pw = torch.clamp(self.weight, min=0)
            nw = torch.clamp(self.weight, max=0)
            px = torch.clamp(self.X, min=0)
            nx = torch.clamp(self.X, max=0)


            # def f(w1, w2, x1, x2):
            #     Z1 = F.conv2d(x1, w1, bias=self.bias, stride=self.stride, padding=self.padding, groups=self.groups)
            #     Z2 = F.conv2d(x2, w2, bias=self.bias, stride=self.stride, padding=self.padding, groups=self.groups)
            #     S1 = safe_divide(R, Z1)
            #     S2 = safe_divide(R, Z2)
            #     C1 = x1 * self.gradprop(Z1, x1, S1)[0]
            #     C2 = x2 * self.gradprop(Z2, x2, S2)[0]
            #      return C1 + C2

            def f(w1, w2, x1, x2):
                Z1 = F.conv2d(x1, w1, bias=self.bias, stride=self.stride, dilation=self.dilation, padding=self.padding, groups=self.groups)
                Z2 = F.conv2d(x2, w2, bias=self.bias, stride=self.stride, dilation=self.dilation, padding=self.padding, groups=self.groups)
                Z = Z1 + Z2
                S = safe_divide(R, Z)
                C1 = x1 * self.gradprop(Z1, x1, S)[0]
                C2 = x2 * self.gradprop(Z2, x2, S)[0]
                return C1 + C2



            activator_relevances = f(pw, nw, px, nx)
            inhibitor_relevances = f(nw, pw, px, nx)

            R = alpha * activator_relevances - beta * inhibitor_relevances
        return R

    def RAP_relprop(self, R_p):
        def shift_rel(R, R_val):
            R_nonzero = torch.ne(R, 0).type(R.type())
            shift = safe_divide(R_val, torch.sum(R_nonzero, dim=[1,2,3], keepdim=True)) * torch.ne(R, 0).type(R.type())
            K = R - shift
            return K
        def pos_prop(R, Za1, Za2, x1):
            R_pos = torch.clamp(R, min=0)
            R_neg = torch.clamp(R, max=0)
            S1 = safe_divide((R_pos * safe_divide((Za1 + Za2), Za1 + Za2)), Za1)
            C1 = x1 * self.gradprop(Za1, x1, S1)[0]
            S1n = safe_divide((R_neg * safe_divide((Za1 + Za2), Za1 + Za2)), Za2)
            C1n = x1 * self.gradprop(Za2, x1, S1n)[0]
            S2 = safe_divide((R_pos * safe_divide((Za2), Za1 + Za2)), Za2)
            C2 = x1 * self.gradprop(Za2, x1, S2)[0]
            S2n = safe_divide((R_neg * safe_divide((Za2), Za1 + Za2)), Za2)
            C2n = x1 * self.gradprop(Za2, x1, S2n)[0]
            Cp = C1 + C2
            Cn = C2n + C1n
            C = (Cp + Cn)
            C = shift_rel(C, C.sum(dim=[1,2,3], keepdim=True) - R.sum(dim=[1,2,3], keepdim=True))
            return C

        def f(R, w1, w2, x1, x2):
            R_nonzero = R.ne(0).type(R.type())
            Za1 = F.conv2d(x1, w1, bias=None, stride=self.stride, padding=self.padding, groups=self.groups) * R_nonzero
            Za2 = - F.conv2d(x1, w2, bias=None, stride=self.stride, padding=self.padding, groups=self.groups) * R_nonzero

            Zb1 = - F.conv2d(x2, w1, bias=None, stride=self.stride, padding=self.padding, groups=self.groups) * R_nonzero
            Zb2 = F.conv2d(x2, w2, bias=None, stride=self.stride, padding=self.padding, groups=self.groups) * R_nonzero

            C1 = pos_prop(R, Za1, Za2, x1)
            C2 = pos_prop(R, Zb1, Zb2, x2)
            return C1 #+ C2

        def backward(R_p, px, nx, pw, nw):

            # if torch.is_tensor(self.bias):
            #     bias = self.bias.unsqueeze(-1).unsqueeze(-1)
            #     bias_p = safe_divide(bias * R_p.ne(0).type(self.bias.type()),
            #                          R_p.ne(0).type(self.bias.type()).sum(dim=[2, 3], keepdim=True))
            #     R_p = R_p - bias_p

            Rp = f(R_p, pw, nw, px, nx)

            # if torch.is_tensor(self.bias):
            #     Bp = f(bias_p, pw, nw, px, nx)
            #
            #     Rp = Rp + Bp
            return Rp
        def final_backward(R_p, pw, nw, X1):
            X = X1
            L = X * 0 + \
                torch.min(torch.min(torch.min(X, dim=1, keepdim=True)[0], dim=2, keepdim=True)[0], dim=3,
                          keepdim=True)[0]
            H = X * 0 + \
                torch.max(torch.max(torch.max(X, dim=1, keepdim=True)[0], dim=2, keepdim=True)[0], dim=3,
                          keepdim=True)[0]
            Za = torch.conv2d(X, self.weight, bias=None, stride=self.stride, padding=self.padding, groups=self.groups) - \
                 torch.conv2d(L, pw, bias=None, stride=self.stride, padding=self.padding, groups=self.groups) - \
                 torch.conv2d(H, nw, bias=None, stride=self.stride, padding=self.padding, groups=self.groups)

            Sp = safe_divide(R_p, Za)

            Rp = X * self.gradprop2(Sp, self.weight) - L * self.gradprop2(Sp, pw) - H * self.gradprop2(Sp, nw)
            return Rp
        pw = torch.clamp(self.weight, min=0)
        nw = torch.clamp(self.weight, max=0)
        px = torch.clamp(self.X, min=0)
        nx = torch.clamp(self.X, max=0)

        if self.X.shape[1] == 3:
            Rp = final_backward(R_p, pw, nw, self.X)
        else:
            Rp = backward(R_p, px, nx, pw, nw)

        return Rp

