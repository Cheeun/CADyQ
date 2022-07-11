# reproduced by Cheeun Hong 

from model import common

import torch
import torch.nn as nn
import math
from model.quant_ops import conv9x9, conv3x3



class SRResNet(nn.Module):
    def __init__(self, args, is_teacher=False, conv=common.default_conv):
        super(SRResNet, self).__init__()


        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3 
        scale = args.scale[0]
        # act = nn.LeakyReLU(0.2, inplace=True)
        act = nn.PReLU()

        self.is_teacher = is_teacher

        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # define head module
        m_head = [conv9x9(args.n_colors, n_feats, kernel_size=9, bias=False)]
        m_head.append(act)

        m_body = [
            common.ResBlock_srresnet(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ] 
        
        m_body.append( conv3x3(n_feats, n_feats, kernel_size, bias=False))
        m_body.append( nn.BatchNorm2d(n_feats))
        # m_body.append(nn.InstanceNorm2d(n_feats, affine=True))

         


        m_tail = [
            common.Upsampler_srresnet(conv3x3, scale, n_feats, act=False),
            conv9x9(n_feats, args.n_colors, kernel_size=9, bias=False)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):

        # x = self.sub_mean(x)
        x = self.head(x)
        
        res = self.body(x)
        res += x

        out = res

        x = self.tail(res)
        # x = self.add_mean(x)
        if self.is_teacher:
            return x, out
        else:
            return x


