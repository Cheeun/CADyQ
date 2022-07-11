import torch
import torch.nn as nn
import torch.nn.functional as F

from model.quant_ops import quant_act_pams, quant_conv3x3

class FBlock_PAMS(nn.Module):
    def __init__(self, conv, num_features, k_bits =32, bias=False, ema_epoch=1, linq=False):
        super(FBlock_PAMS, self).__init__()

        self.quant_actf1 = quant_act_pams(k_bits, ema_epoch=ema_epoch)
        self.quant_actf2 = quant_act_pams(k_bits, ema_epoch=ema_epoch)

        self.module = nn.Sequential(
            self.quant_actf1,
            conv(3, num_features, kernel_size=3,  k_bits=k_bits, bias=bias),
            nn.LeakyReLU(0.05),
            self.quant_actf2,
            conv(num_features, num_features, kernel_size=3,  k_bits=k_bits, bias=bias),
            nn.LeakyReLU(0.05)
        )

    def forward(self, x):
        return self.module(x) 

class DBlock_PAMS(nn.Module):
    def __init__(self, conv, num_features, d, s, bias=False, k_bits=32, ema_epoch=1, loss_kdf=False, linq=False):
        super(DBlock_PAMS, self).__init__()
        self.num_features = num_features
        self.s = s
        self.k_bits = k_bits 

        self.quant_act1 = quant_act_pams(self.k_bits,ema_epoch=ema_epoch) 
        self.quant_act2 = quant_act_pams(self.k_bits,ema_epoch=ema_epoch)
        self.quant_act3 = quant_act_pams(self.k_bits,ema_epoch=ema_epoch)
        self.quant_act4 = quant_act_pams(self.k_bits,ema_epoch=ema_epoch) 
        self.quant_act5 = quant_act_pams(self.k_bits,ema_epoch=ema_epoch)
        self.quant_act6 = quant_act_pams(self.k_bits,ema_epoch=ema_epoch)


        self.enhancement_top = nn.Sequential(
            self.quant_act1,
            conv(num_features, num_features - d, kernel_size=3, k_bits=self.k_bits, bias=bias),
            nn.LeakyReLU(0.05),
            self.quant_act2, 
            conv(num_features - d, num_features - 2 * d, kernel_size=3, k_bits=self.k_bits, bias=bias, groups=4),
            nn.LeakyReLU(0.05),
            self.quant_act3,
            conv(num_features - 2 * d, num_features, kernel_size=3, k_bits=self.k_bits, bias=bias),
            nn.LeakyReLU(0.05)
        )
        self.enhancement_bottom = nn.Sequential(
            self.quant_act4,
            conv(num_features - d, num_features, kernel_size=3, k_bits=self.k_bits, bias=bias),
            nn.LeakyReLU(0.05),
            self.quant_act5,
            conv(num_features, num_features - d, kernel_size=3, k_bits=self.k_bits, bias=bias, groups=4),
            nn.LeakyReLU(0.05),
            self.quant_act6,
            conv(num_features - d, num_features + d, kernel_size=3, k_bits=self.k_bits, bias=bias),
            nn.LeakyReLU(0.05)
        )
        self.compression = nn.Conv2d(num_features + d, num_features, kernel_size=1)
        self.loss_kdf= loss_kdf

    def forward(self, x):
        f = x[1]
        x = x[0]

        residual = x
        x = self.enhancement_top(x)

        slice_1 = x[:, :int((self.num_features - self.num_features/self.s)), :, :]
        slice_2 = x[:, int((self.num_features - self.num_features/self.s)):, :, :]

        x = self.enhancement_bottom(slice_1)
        f1 = x

        x = x + torch.cat((residual, slice_2), 1)
        x = self.compression(x)

        if self.loss_kdf:
            if f is None:
                f = f1.unsqueeze(0)
            else: 
                f = torch.cat([f, f1.unsqueeze(0)], dim=0)
        else:
            f = None

        return [x, f]


class IDN_PAMS(nn.Module):
    def __init__(self, args, conv=quant_conv3x3, bias=False, k_bits=32, linq=False):
        super(IDN_PAMS, self).__init__()
        self.scale = args.scale[0]
        num_features = args.n_feats
        d = args.idn_d
        s = args.idn_s

        self.fblock = FBlock_PAMS(conv, num_features, bias=bias, k_bits=args.k_bits, ema_epoch=args.ema_epoch, linq=linq)

        
        m_dblocks = [
            DBlock_PAMS(conv, num_features, d, s, bias=bias, k_bits=args.k_bits, ema_epoch=args.ema_epoch,loss_kdf=args.loss_kdf, linq=linq) for _ in range(4)

        ]
        self.dblocks = nn.Sequential(*m_dblocks)
        self.deconv = nn.ConvTranspose2d(num_features, 3, kernel_size=17, stride=self.scale, padding=8, output_padding=1)


    def forward(self, x):
        bicubic = F.interpolate(x, scale_factor=self.scale, mode='bicubic', align_corners=False)
        x= self.fblock(x)
        f = None
        x,f = self.dblocks([x,f])
        out = x
        x = self.deconv(x, output_size=bicubic.size())
        
        return bicubic + x, out, f

