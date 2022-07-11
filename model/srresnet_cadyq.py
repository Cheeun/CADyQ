import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia as K

from model import common
from model.quant_ops import conv3x3, conv9x9, quant_conv3x3, quant_conv9x9
from model.cadyq import BitSelector

class ResBlock_CADyQ(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=False, 
                bn=False, inn=False, act=nn.PReLU(), res_scale=1, k_bits = 32, ema_epoch=1, search_space=[2,4,8], loss_kdf=False, linq=False):

        super(ResBlock_CADyQ, self).__init__()
        self.k_bits = k_bits

        self.classify1 = BitSelector(n_feats, bias=bias, ema_epoch=ema_epoch, search_space=search_space, linq=linq)
        self.classify2 = BitSelector(n_feats, bias=bias, ema_epoch=ema_epoch, search_space=search_space, linq=linq)

        self.conv1 = conv(n_feats, n_feats, kernel_size, k_bits=self.k_bits, bias=bias)
        self.conv2 = conv(n_feats, n_feats, kernel_size, k_bits=self.k_bits, bias=bias)
        self.bn1 = nn.BatchNorm2d(n_feats)
        # self.bn1 = nn.InstanceNorm2d(n_feats, affine=True)

        self.act = act
        self.bn2 = nn.BatchNorm2d(n_feats)
        # self.bn2 = nn.InstanceNorm2d(n_feats, affine=True)

        self.res_scale = res_scale

        self.shortcut = common.ShortCut()
        self.loss_kdf = loss_kdf


    def forward(self, x):
        weighted_bits = x[4]
        f = x[3]
        bits = x[2]
        grad = x[0]
        x = x[1]

        grad,residual,bits,weighted_bits = self.classify1([grad, self.shortcut(x),bits,weighted_bits]) 
        res = self.act(self.bn1(self.conv1(x))) 

        grad, res, bits, weighted_bits = self.classify2([grad, res, bits, weighted_bits])
        res = self.conv2(res)
        f1 = res
        res = self.bn2(res).mul(self.res_scale)
        res += residual
        if self.loss_kdf:
            if f is None:
                f = f1.unsqueeze(0)
            else: 
                f = torch.cat([f, f1.unsqueeze(0)], dim=0)
        else:
            f = None

        return [grad, res, bits, f, weighted_bits]



class SRResNet_CADyQ(nn.Module):
    def __init__(self, args, conv=quant_conv3x3, bias=False, k_bits =32 ):
        super(SRResNet_CADyQ, self).__init__()


        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3 
        scale = args.scale[0]
        act = nn.PReLU()
        # act = nn.LeakyReLU(0.2, inplace=True)

        self.fully = args.fully
        self.k_bits = k_bits

        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)


        m_head = [conv9x9(args.n_colors, n_feats, kernel_size=9, bias=False)]
        m_head.append(act)

        m_body = [
            ResBlock_CADyQ(
                quant_conv3x3, n_feats, kernel_size, bn=True, act=act, res_scale=args.res_scale, k_bits=self.k_bits, bias=bias, ema_epoch=args.ema_epoch, search_space=args.search_space, loss_kdf=args.loss_kdf, linq=args.linq
            ) for _ in range(n_resblocks)
        ]
        m_body.append( conv3x3(n_feats, n_feats, kernel_size, bias=False))
        m_body.append( nn.BatchNorm2d(n_feats))
        
        m_tail = [
            common.Upsampler_srresnet(conv3x3, scale, n_feats, act=False),
            conv9x9(n_feats, args.n_colors, kernel_size=9, bias=False)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)


    def forward(self, x):

        # x = self.sub_mean(x)
        image = x
        grads: torch.Tensor = K.filters.spatial_gradient(K.color.rgb_to_grayscale(image/255.), order=1)
        image_grad = torch.mean(torch.abs(grads.squeeze(1)),(2,3)) *1e+3 # abs version
        if self.fully: x = self.quant_head(x)

        x = self.head(x)
        res = x
        bits = 0; weighted_bits=0; f=None
        
        image_grad, res, bits, f, weighted_bits = self.body[0:-2]([image_grad, res, bits, f, weighted_bits])

        res = self.body[-2:](res)
        res += x
        out = res
        x = self.tail(res)
        # x = self.add_mean(x)

        return x, out, bits, f, weighted_bits


