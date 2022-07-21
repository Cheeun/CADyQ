import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia as K

from model import common
from model.quant_ops import conv3x3, quant_conv3x3, quant_act_pams
from model.cadyq import BitSelector


class ResidualBlock_CADyQ(nn.Module):
    def __init__(self, 
                 in_channels, out_channels, conv, act, kernel_size, res_scale, k_bits=32, bias=False, ema_epoch=1,search_space=[4,6,8],loss_kdf=False, linq=False):
        super(ResidualBlock_CADyQ, self).__init__()

        self.bitsel1 = BitSelector(in_channels, bias=bias, ema_epoch=ema_epoch, search_space=search_space)

        self.body = nn.Sequential(
            conv(in_channels, out_channels, k_bits=k_bits, bias=bias, kernel_size=kernel_size, stride=1, padding=1),
            act,
            BitSelector(out_channels, bias=bias, ema_epoch=ema_epoch, search_space=search_space),
            conv(out_channels, out_channels, k_bits=k_bits, bias=bias, kernel_size=kernel_size, stride=1, padding=1),
        )
        self.loss_kdf= loss_kdf
        self.res_scale = res_scale

        self.quant_act3 = quant_act_pams(k_bits, ema_epoch=ema_epoch)
        self.shortcut = common.ShortCut()



    def forward(self, x):
        weighted_bits = x[4]
        f = x[3]
        bits = x[2]
        grad = x[0]
        x = x[1]

        x = self.shortcut(x)
        grad,x,bits,weighted_bits = self.bitsel1([grad,x,bits,weighted_bits]) # cadyq
        residual = x
        # grad,x,bits,weighted_bits= self.body[0]() # cadyq
        # x = self.body[1:3](x) # conv-relu
        x = self.body[0:2](x) # conv-relu
        # grad,x,bits,weighted_bits= self.body[3]([grad,x,bits,weighted_bits]) # cadyq
        grad,x,bits,weighted_bits= self.body[2]([grad,x,bits,weighted_bits]) # cadyq
        # out = self.body[4](x) # conv
        out = self.body[3](x) # conv
        f1 = out
        out = out.mul(self.res_scale)
        out = self.quant_act3(out)
        out += residual
        if self.loss_kdf:
            if f is None:
                f = f1.unsqueeze(0)
            else: 
                f = torch.cat([f, f1.unsqueeze(0)], dim=0)
        else:
            f = None


        return [grad, out, bits, f, weighted_bits]




class EDSR_CADyQ(nn.Module):
    def __init__(self, args, conv=quant_conv3x3, bias = False, k_bits = 32): 
        super(EDSR_CADyQ, self).__init__()

        n_resblock = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3 
        scale = args.scale[0]
        act = nn.ReLU(True)
        self.k_bits = k_bits

        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)
        device = torch.device('cpu' if args.cpu else f'cuda:{args.gpu_id}')


        m_head = [conv3x3(args.n_colors, n_feats, kernel_size,  bias=bias)]

        # baseline= (n_resblock == 16 )
        m_body = [
            ResidualBlock_CADyQ(n_feats, n_feats, quant_conv3x3, act, kernel_size, res_scale=args.res_scale, k_bits=self.k_bits, bias=bias, ema_epoch=args.ema_epoch, search_space=args.search_space, loss_kdf=args.loss_kdf, linq=args.linq
            ) for i in range(n_resblock)
        ]
        m_body.append(conv3x3(n_feats, n_feats, kernel_size, bias= bias))


        m_tail = [
            common.Upsampler(conv3x3, scale, n_feats, act=False),
            nn.Conv2d(n_feats, args.n_colors, kernel_size,padding=(kernel_size//2))
        ]
        
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):

        image = x
        grads: torch.Tensor = K.filters.spatial_gradient(K.color.rgb_to_grayscale(image/255.), order=1)
        image_grad = torch.mean(torch.abs(grads.squeeze(1)),(2,3)) *1e+3 # [16,2]
        
        f=None; weighted_bits = 0; bits=0; 

        x = self.sub_mean(x)

        x = self.head(x)
        res = x

        image_grad, res, bits, f, weighted_bits = self.body[:-1]([image_grad, res, bits, f,weighted_bits])
        res = self.body[-1](res)

        res += x
        out = res
        x = self.tail(res)

        x = self.add_mean(x)

        return x, out, bits, f, weighted_bits

