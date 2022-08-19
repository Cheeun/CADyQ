# added by Cheeun on 21.10.24 
# from https://github.com/nmhkahn/CARN-pytorch/blob/master/carn/model/carn.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model.quant_ops import quant_act_pams

def init_weights(modules):
    pass
   
class MeanShift(nn.Module):
    def __init__(self, mean_rgb, sub):
        super(MeanShift, self).__init__()

        sign = -1 if sub else 1
        r = mean_rgb[0] * sign
        g = mean_rgb[1] * sign
        b = mean_rgb[2] * sign

        self.shifter = nn.Conv2d(3, 3, 1, 1, 0)
        self.shifter.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.shifter.bias.data   = torch.Tensor([r, g, b])

        # Freeze the mean shift layer
        for params in self.shifter.parameters():
            params.requires_grad = False

    def forward(self, x):
        x = self.shifter(x)
        return x



class UpsampleBlock(nn.Module):
    def __init__(self, 
                 n_channels, scale, multi_scale, group=1, fully=False, k_bits =32):
        super(UpsampleBlock, self).__init__()

        if multi_scale:
            self.up2 = _UpsampleBlock(n_channels, scale=2, group=group, fully=fully, k_bits=k_bits)
            self.up3 = _UpsampleBlock(n_channels, scale=3, group=group, fully=fully, k_bits=k_bits)
            self.up4 = _UpsampleBlock(n_channels, scale=4, group=group, fully=fully, k_bits=k_bits)
        else:
            self.up =  _UpsampleBlock(n_channels, scale=scale, group=group, fully=fully, k_bits=k_bits)
            # self.up4 =  _UpsampleBlock(n_channels, scale=scale, group=group)


        self.multi_scale = multi_scale

    def forward(self, x, scale):
    # def forward(self, x, scale=4):

        if self.multi_scale:
            if scale == 2:
                return self.up2(x)
            elif scale == 3:
                return self.up3(x)
            elif scale == 4:
                return self.up4(x)
        else:
            return self.up(x)


class _UpsampleBlock(nn.Module):
    def __init__(self, 
				 n_channels, scale, 
				 group=1, fully=False, k_bits =32):
        super(_UpsampleBlock, self).__init__()

        modules = []
        if scale == 2 or scale == 4 or scale == 8:
            for _ in range(int(math.log(scale, 2))):
                if fully: 
                    # modules += [quant_act_lin(k_bits)]
                    modules += [quant_act_pams(k_bits, ema_epoch=1)]

                modules += [nn.Conv2d(n_channels, 4*n_channels, 3, 1, 1, groups=group), nn.ReLU(inplace=True)]
                modules += [nn.PixelShuffle(2)]
        elif scale == 3:
            modules += [nn.Conv2d(n_channels, 9*n_channels, 3, 1, 1, groups=group), nn.ReLU(inplace=True)]
            modules += [nn.PixelShuffle(3)]

        self.body = nn.Sequential(*modules)
        init_weights(self.modules)
        
    def forward(self, x):
        out = self.body(x)
        return out

class BasicBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 ksize=3, stride=1, pad=1):
        super(BasicBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
            nn.ReLU(inplace=True)
        )

        init_weights(self.modules)
        
    def forward(self, x):
        out = self.body(x)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, 
                 in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )

        init_weights(self.modules)
        
    def forward(self, x):
        f = x[1]
        x = x[0]


        out = self.body(x)
        f1 = out
        out = F.relu(out + x)
        if f is None:
            f = f1.unsqueeze(0)
        else: 
            f = torch.cat([f, f1.unsqueeze(0)], dim=0)
        # return out
        return [out, f]

class CARNBlock(nn.Module):
    def __init__(self, 
                 in_channels, out_channels,
                 group=1):
        super(CARNBlock, self).__init__()

        self.b1 = ResidualBlock(64, 64)
        self.b2 = ResidualBlock(64, 64)
        self.b3 = ResidualBlock(64, 64)
        self.c1 = BasicBlock(64*2, 64, 1, 1, 0)
        self.c2 = BasicBlock(64*3, 64, 1, 1, 0)
        self.c3 = BasicBlock(64*4, 64, 1, 1, 0)

    def forward(self, x):
        # added for teacher
        f = x[1]
        x = x[0]

        c0 = o0 = x
        b1,f = self.b1([o0,f])
        # b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)

        b2,f = self.b2([o1,f])
        # b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)

        b3,f = self.b3([o2,f])
        # b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        return [o3, f]
        # return o3

class CARN(nn.Module):
    def __init__(self, args, is_teacher=False, multi_scale=False):
        super(CARN, self).__init__()
        
        scale = args.scale[0]
        # multi_scale = args.multi_scale
        
        group = args.group 

        self.is_teacher =is_teacher

        self.sub_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=False)
        
        self.entry = nn.Conv2d(3, 64, 3, 1, 1)

        self.b1 = CARNBlock(64, 64)
        self.b2 = CARNBlock(64, 64)
        self.b3 = CARNBlock(64, 64)
        self.c1 = BasicBlock(64*2, 64, 1, 1, 0)
        self.c2 = BasicBlock(64*3, 64, 1, 1, 0)
        self.c3 = BasicBlock(64*4, 64, 1, 1, 0)
        
        self.upsample = UpsampleBlock(64, scale=scale, multi_scale=multi_scale, group=group)
        self.exit = nn.Conv2d(64, 3, 3, 1, 1)
                
    def forward(self, x, scale):
    # def forward(self, x, scale=4):
        # import pdb; pdb.set_trace()
        x = self.sub_mean(x)
        x = self.entry(x)
        c0 = o0 = x

        if self.is_teacher:
            f = None
            b1,f = self.b1([o0,f])
        else:
            b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)
        
        if self.is_teacher:
            b2,f = self.b2([o1,f])
        else:
            b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)
        
        if self.is_teacher:
            b3,f = self.b3([o2,f])
        else:
            b3 = self.b3(o2)

        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        feat = o3
        
        out = self.upsample(o3, scale=scale)

        out = self.exit(out)
        out = self.add_mean(out)

        if self.is_teacher:
            return out, feat, f
        else:
            return out
        