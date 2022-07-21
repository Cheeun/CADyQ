import torch
import torch.nn as nn
import torch.nn.functional as F

from model.quant_ops import quant_act_pams, quant_conv3x3, conv3x3
from model import common

class ResBlock_PAMS(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=False, 
                bn=False, act=nn.ReLU(False), res_scale=1, k_bits = 32, ema_epoch=1, loss_kdf=False):

        super(ResBlock_PAMS, self).__init__()
        self.k_bits = k_bits

        self.quant_act1 = quant_act_pams(self.k_bits,ema_epoch=ema_epoch)
        self.quant_act2 = quant_act_pams(self.k_bits, ema_epoch=ema_epoch)
        self.quant_act3 = quant_act_pams(self.k_bits, ema_epoch=ema_epoch)
        
        self.shortcut = common.ShortCut()


        self.body = nn.Sequential(
            # self.quant_act1,
            conv(n_feats, n_feats, k_bits=k_bits, bias=bias, kernel_size=3, stride=1, padding=1),
            act,
            self.quant_act2,
            conv(n_feats, n_feats, k_bits=k_bits, bias=bias, kernel_size=3, stride=1, padding=1),
        )

        self.res_scale = res_scale
        self.loss_kdf = loss_kdf

    def forward(self, x):
        f = x[1]
        x = x[0]
        
        # residual = self.body[0](self.shortcut(x))
        # body = self.body[1:](residual) 
        residual = self.quant_act1(self.shortcut(x))
        body = self.body(residual) 
        
        f2 = body 
        body = body.mul(self.res_scale)
        res = self.quant_act3(body)
        # res = body
        res += residual

        if self.loss_kdf:
            new_f = f2.unsqueeze(0) 
            if f is None:
                f = new_f
            else: 
                f = torch.cat([f, new_f], dim=0)
        else:
            f = None

        return [res, f]

class EDSR_PAMS(nn.Module):
    def __init__(self, args, conv=quant_conv3x3, bias = False, k_bits = 32, mixed=False,linq=False,fully=False):
        super(EDSR_PAMS, self).__init__()

        n_resblock = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3 
        scale = args.scale[0]
        act = nn.ReLU(True)
        self.k_bits = args.k_bits
        self.fully = fully

        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        m_head = [conv3x3(args.n_colors, n_feats, kernel_size,  bias=bias)]

        # baseline = (n_resblock == 16)
        m_body = [
                ResBlock_PAMS(
                    quant_conv3x3, n_feats, kernel_size, act=act, res_scale=args.res_scale, k_bits=self.k_bits, bias = bias, ema_epoch=args.ema_epoch, loss_kdf=args.loss_kdf, 
                ) for i in range(n_resblock)
            ]
        m_body.append(conv3x3(n_feats, n_feats, kernel_size, bias= bias))


        m_tail = [
            common.Upsampler(conv3x3, scale, n_feats, act=False),
            conv3x3(n_feats, args.n_colors, kernel_size, bias=bias)
            # nn.Conv2d(n_feats, args.n_colors, kernel_size, padding=(kernel_size//2))
        ]
        
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        
        f=None

        res,f = self.body[:-1]([x,f])
        res = self.body[-1](res)
        res += x

        out = res
        x = self.tail(res)
        x = self.add_mean(x)

        return x, out, f

    
    @property
    def name(self):
        return 'edsr'