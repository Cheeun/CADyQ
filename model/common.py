import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.quant_ops import quant_act_pams, quant_act_lin


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class ShortCut(nn.Module):
    def __init__(self):
        super(ShortCut, self).__init__()

    def forward(self, input):
        return input

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std).cuda()
        # self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1).cuda() / std.view(3, 1, 1, 1)

        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean).cuda() / std
        for p in self.parameters():
            p.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, inn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            elif inn:
                m.append(nn.InstanceNorm2d(n_feats, affine=True))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale
        self.shortcut = ShortCut()

    def forward(self, x):
        residual = self.shortcut(x)
        # print("hi")
        res = self.body(x).mul(self.res_scale)
        res += residual

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))
                elif act == 'lrelu':
                    m.append(nn.LeakyReLU(0.2, inplace=True))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
            elif act == 'lrelu':
                    m.append(nn.LeakyReLU(0.2, inplace=True))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)



class Upsampler_linq(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True, k_bits=32, ema_epoch=1):
        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(quant_act_pams(k_bits, ema_epoch=ema_epoch))
                # m.append(quant_act_lin(k_bits))
                m.append(conv(n_feats, 4 * n_feats, kernel_size=3, bias=bias, k_bits = k_bits))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        super(Upsampler_linq, self).__init__(*m)

class Upsampler_q(nn.Module):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True, k_bits=32, ema_epoch=1, search_space=[4,6,8]):
        super(Upsampler_q, self).__init__()
        
        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                # m.append(quant_act_pams(k_bits, ema_epoch=ema_epoch))
                m.append(classify(n_feats, bias=bias, ema_epoch=ema_epoch, search_space=search_space))
                m.append(conv(n_feats, 4 * n_feats, kernel_size=3, bias=bias, k_bits = k_bits))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        self.m = nn.Sequential(*m)

    def forward(self,x):
        weighted_bits = x[3]
        # f = x[3]
        bits = x[2]
        grad = x[0]
        x = x[1]

        # check if the pretrained model's name is like this
        # return self.m(x)
        grad, x, bits, weighted_bits = self.m[0]([grad,x,bits,weighted_bits])
        x= self.m[1:3](x)
        grad, x, bits, weighted_bits = self.m[3]([grad,x,bits,weighted_bits])
        x= self.m[4:6](x)

        return [grad, x, bits, weighted_bits]
        
        


class ResBlock_srresnet(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=False, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock_srresnet, self).__init__()

        self.conv1 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.conv2 = conv(n_feats, n_feats, kernel_size, bias=bias)

        self.bn1 = nn.BatchNorm2d(n_feats)
        self.act = act
        self.bn2 = nn.BatchNorm2d(n_feats)
        self.res_scale = res_scale


        self.res_scale = res_scale
        self.shortcut = ShortCut()

    def forward(self, x):
        residual = self.shortcut(x)
        res = self.act(self.bn1(self.conv1(x)))
        res = self.bn2(self.conv2(res)).mul(self.res_scale)
        res += residual


        # residual = self.shortcut(x)
        # res = self.body(x).mul(self.res_scale)
        # res += residual

        return res

class Upsampler_srresnet(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
        # scale = 4 # for SRResNet
        m = []
        if scale == 4:
            m.append(conv(n_feats, 4 * n_feats, 3, bias=False))
            m.append(nn.PixelShuffle(2))
            m.append(nn.PReLU())
            # m.append(nn.LeakyReLU(0.2, inplace=True))
            m.append(conv(n_feats, 4 * n_feats, 3, bias=False))
            m.append(nn.PixelShuffle(2))
            # m.append(nn.LeakyReLU(0.2, inplace=True))
            m.append(nn.PReLU())
        elif scale ==2 :
            m.append(conv(n_feats, 4 * n_feats, 3, bias=False))
            m.append(nn.PixelShuffle(2))
            m.append(nn.PReLU())
        else:
            print("not implemented")
        

        super(Upsampler_srresnet, self).__init__(*m)


class Upsampler_srresnet_linq(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True, k_bits=32):

        scale = 4 # for SRResNet
        m = []
        m.append(quant_act_lin(k_bits))
        m.append(conv(n_feats, 4 * n_feats, 3, bias=False))
        m.append(nn.PixelShuffle(2))
        m.append(nn.PReLU())
        # m.append(nn.LeakyReLU(0.2, inplace=True))
        m.append(quant_act_lin(k_bits))
        m.append(conv(n_feats, 4 * n_feats, 3, bias=False))
        m.append(nn.PixelShuffle(2))
        # m.append(nn.LeakyReLU(0.2, inplace=True))
        m.append(nn.PReLU())
        

        super(Upsampler_srresnet_linq, self).__init__(*m)
