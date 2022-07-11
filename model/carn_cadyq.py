import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia as K

from model import carn, carn_pams
from model.quant_ops import quant_conv3x3
from model.cadyq import BitSelector

class ResidualBlock_CADyQ(nn.Module):
    def __init__(self, 
                 in_channels, out_channels,  conv, k_bits=32, bias=False, ema_epoch=1,search_space=[4,6,8],loss_kdf=False, linq=False):
        super(ResidualBlock_CADyQ, self).__init__()

        self.body = nn.Sequential(
            BitSelector(in_channels, bias=bias, ema_epoch=ema_epoch, search_space=search_space),
            conv(in_channels, out_channels, k_bits=k_bits, bias=bias, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            BitSelector(out_channels, bias=bias, ema_epoch=ema_epoch, search_space=search_space),
            conv(out_channels, out_channels, k_bits=k_bits, bias=bias, kernel_size=3, stride=1, padding=1),
        )
        self.loss_kdf= loss_kdf

    def forward(self, x):
        weighted_bits = x[4]
        f = x[3]
        bits = x[2]
        grad = x[0]
        x = x[1]

        residual = x
        grad,x,bits,weighted_bits= self.body[0]([grad,x,bits,weighted_bits]) # cadyq
        x = self.body[1:3](x) # conv-relu
        grad,x,bits,weighted_bits= self.body[3]([grad,x,bits,weighted_bits]) # cadyq
        out = self.body[4](x) # conv
        f1 = out
        if self.loss_kdf:
            if f is None:
                f = f1.unsqueeze(0)
            else: 
                f = torch.cat([f, f1.unsqueeze(0)], dim=0)
        else:
            f = None

        out = F.relu(out + residual)
        return [grad, out, bits, f, weighted_bits]


class CARNBlock_CADyQ(nn.Module):
    def __init__(self, 
                 in_channels, out_channels, conv, k_bits=32, bias=False, ema_epoch=1, group=1, search_space=[4,6,8],loss_kdf=False, linq=False, fully=False):
        super(CARNBlock_CADyQ, self).__init__()

        self.b1 = ResidualBlock_CADyQ(in_channels, out_channels, conv, k_bits=k_bits, bias=bias, ema_epoch=ema_epoch,search_space=search_space,loss_kdf=loss_kdf,linq=linq)
        self.b2 = ResidualBlock_CADyQ(in_channels, out_channels, conv, k_bits=k_bits, bias=bias, ema_epoch=ema_epoch,search_space=search_space,loss_kdf=loss_kdf,linq=linq)
        self.b3 = ResidualBlock_CADyQ(in_channels, out_channels, conv, k_bits=k_bits, bias=bias, ema_epoch=ema_epoch,search_space=search_space,loss_kdf=loss_kdf,linq=linq)

        if fully:
            self.c1 = carn_pams.PAMS_BasicBlock(in_channels*2, out_channels, conv, k_bits=k_bits, bias=bias,ema_epoch=ema_epoch, ksize=1, stride=1, pad=0)
            self.c2 = carn_pams.PAMS_BasicBlock(in_channels*3, out_channels, conv, k_bits=k_bits, bias=bias,ema_epoch=ema_epoch, ksize=1, stride=1, pad=0)
            self.c3 = carn_pams.PAMS_BasicBlock(in_channels*4, out_channels, conv, k_bits=k_bits, bias=bias,ema_epoch=ema_epoch, ksize=1, stride=1, pad=0)
        else:
            self.c1 = carn.BasicBlock(in_channels*2, out_channels, 1, 1, 0)
            self.c2 = carn.BasicBlock(in_channels*3, out_channels, 1, 1, 0)
            self.c3 = carn.BasicBlock(in_channels*4, out_channels, 1, 1, 0)
    

    def forward(self, x):
        weighted_bits = x[4]
        f = x[3]
        bits = x[2]
        grad = x[0]
        x = x[1]

        c0 = o0 = x

        grad, b1, bits, f, weighted_bits = self.b1([grad, o0, bits, f, weighted_bits])
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)

        grad, b2, bits, f, weighted_bits = self.b2([grad, o1, bits, f, weighted_bits])
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)
        
        grad, b3, bits, f, weighted_bits = self.b3([grad, o2, bits, f, weighted_bits])
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        return [grad, o3, bits, f, weighted_bits]

class CARN_CADyQ(nn.Module):
    def __init__(self, args, conv=quant_conv3x3, bias=False, k_bits=32, multi_scale=False):
        super(CARN_CADyQ, self).__init__()
        
        scale = args.scale[0]
        group = args.group 
        n_feats = args.n_feats 
        self.fully = args.fully

        self.k_bits = args.k_bits

        self.sub_mean = carn.MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = carn.MeanShift((0.4488, 0.4371, 0.4040), sub=False)
        

        self.entry = nn.Conv2d(3, n_feats, 3, 1, 1)

        self.b1 = CARNBlock_CADyQ(n_feats, n_feats, conv, k_bits=args.k_bits, bias=bias,ema_epoch=args.ema_epoch,search_space=args.search_space,loss_kdf=args.loss_kdf,linq=args.linq,fully=args.fully)
        self.b2 = CARNBlock_CADyQ(n_feats, n_feats, conv, k_bits=args.k_bits, bias=bias,ema_epoch=args.ema_epoch,search_space=args.search_space,loss_kdf=args.loss_kdf,linq=args.linq,fully=args.fully)
        self.b3 = CARNBlock_CADyQ(n_feats, n_feats, conv, k_bits=args.k_bits, bias=bias,ema_epoch=args.ema_epoch,search_space=args.search_space,loss_kdf=args.loss_kdf,linq=args.linq,fully=args.fully)
        
        self.c1 = carn.BasicBlock(n_feats*2, n_feats, 1, 1, 0)
        self.c2 = carn.BasicBlock(n_feats*3, n_feats, 1, 1, 0)
        self.c3 = carn.BasicBlock(n_feats*4, n_feats, 1, 1, 0)
        
        self.upsample = carn.UpsampleBlock(n_feats, scale=scale, multi_scale=multi_scale, group=group,  fully=args.fully, k_bits=args.k_bits)
        self.exit = nn.Conv2d(n_feats, 3, 3, 1, 1)



    def forward(self, x, scale):
        image = x
        grads: torch.Tensor = K.filters.spatial_gradient(K.color.rgb_to_grayscale(image), order=1)
        grad = torch.mean(torch.abs(grads.squeeze(1)),(2,3)) # [16,2] # CARN

        f= None; weighted_bits=0; bits=0


        x = self.sub_mean(x) 
        # if self.fully: x = self.quant_head(x)

        x = self.entry(x)

        c0 = o0 = x
        grad, b1, bits, f, weighted_bits = self.b1([grad, o0, bits, f, weighted_bits]) 
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)

        grad, b2, bits, f, weighted_bits = self.b2([grad, o1, bits, f, weighted_bits])
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)
        
        grad, b3, bits, f, weighted_bits = self.b3([grad, o2, bits, f, weighted_bits])
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        feat = o3
        
        out = self.upsample(o3, scale=scale)

        # if self.fully: out = self.quant_tail(out)

        out = self.exit(out)
        out = self.add_mean(out)


        return out, feat, bits, f, weighted_bits

