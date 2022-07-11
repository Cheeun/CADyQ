import torch
import torch.nn as nn
import torch.nn.functional as F

from model import common

from model.quant_ops import quant_act_pams
from model.quant_ops import conv3x3, conv9x9, quant_conv3x3, quant_conv9x9

class ResBlock_PAMS(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=False, 
                bn=False, inn=False, act=nn.PReLU(), res_scale=1, k_bits = 32, ema_epoch=1, name=None, loss_kdf=False, linq=False):

        super(ResBlock_PAMS, self).__init__()
        self.k_bits = k_bits

        self.quant_act1 = quant_act_pams(k_bits,ema_epoch=ema_epoch)
        self.quant_act2 = quant_act_pams(k_bits,ema_epoch=ema_epoch)

        self.conv1 = conv(n_feats, n_feats, kernel_size, k_bits=self.k_bits, bias=bias)
        self.conv2 = conv(n_feats, n_feats, kernel_size, k_bits=self.k_bits, bias=bias)
        self.bn1 = nn.BatchNorm2d(n_feats)

        self.act = act
        self.bn2 = nn.BatchNorm2d(n_feats)

        self.res_scale = res_scale

        self.shortcut = common.ShortCut()
        self.loss_kdf = loss_kdf

            

    def forward(self, x):
        f = x[1]
        x = x[0]

        residual = self.quant_act1(self.shortcut(x))
        res = self.act(self.bn1(self.conv1(x)))

        res = self.quant_act2(res)
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

        return [res, f]

class SRResNet_PAMS(nn.Module):
    def __init__(self, args, conv=quant_conv3x3, bias=False, k_bits =32, linq=False, fully=False):
        super(SRResNet_PAMS, self).__init__()

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3 
        scale = args.scale[0]
        act = nn.PReLU()
        self.fully = fully

        self.k_bits = k_bits

        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        m_head = [conv9x9(args.n_colors, n_feats, kernel_size=9, bias=False)]
        m_head.append(act)

        m_body = [
            ResBlock_PAMS(
                quant_conv3x3, n_feats, kernel_size, bn=True, act=act, res_scale=args.res_scale, k_bits=self.k_bits, bias=bias, ema_epoch=args.ema_epoch, loss_kdf=args.loss_kdf, linq=linq
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
        if self.fully: x = self.quant_head(x)
        x = self.head(x)
        if self.fully: x = self.quant_head2(x)
        f= None

        res,f = self.body[0:-2]([x,f])
        res = self.body[-2:](res)

        res += x
        out = res
        x = self.tail(res)
        # x = self.add_mean(x)

        return x, out, f
