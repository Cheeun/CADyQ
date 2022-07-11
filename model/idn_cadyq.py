import torch
from torch import nn
import torch.nn.functional as F
import kornia as K

from model import idn_pams
from model.quant_ops import quant_conv3x3
from model.cadyq import BitSelector

class DBlock_CADyQ(nn.Module):
    def __init__(self, conv, num_features, d, s, bias=False, k_bits=32, ema_epoch=1, search_space=[2,4,8],loss_kdf=False, linq=False):
        super(DBlock_CADyQ, self).__init__()
        self.num_features = num_features
        self.s = s
        self.k_bits = k_bits 

        self.enhancement_top = nn.Sequential(
            BitSelector(num_features, bias=bias, ema_epoch=ema_epoch, search_space=search_space, linq=linq),
            conv(num_features, num_features - d, kernel_size=3, k_bits=self.k_bits, bias=bias),
            nn.LeakyReLU(0.05),
            BitSelector(num_features - d, bias=bias, ema_epoch=ema_epoch, search_space=search_space, linq=linq),
            conv(num_features - d, num_features - 2 * d, kernel_size=3, k_bits=self.k_bits, bias=bias, groups=4),
            nn.LeakyReLU(0.05),
            BitSelector(num_features - 2 * d, bias=bias, ema_epoch=ema_epoch, search_space=search_space, linq=linq),
            conv(num_features - 2 * d, num_features, kernel_size=3, k_bits=self.k_bits, bias=bias),
            nn.LeakyReLU(0.05)
        )
        self.enhancement_bottom = nn.Sequential(
            BitSelector(num_features - d, bias=bias, ema_epoch=ema_epoch, search_space=search_space, linq=linq),
            conv(num_features - d, num_features, kernel_size=3, k_bits=self.k_bits, bias=bias),
            nn.LeakyReLU(0.05),
            BitSelector(num_features, bias=bias, ema_epoch=ema_epoch, search_space=search_space, linq=linq),
            conv(num_features, num_features - d, kernel_size=3, k_bits=self.k_bits, bias=bias, groups=4),
            nn.LeakyReLU(0.05),
            BitSelector(num_features - d, bias=bias, ema_epoch=ema_epoch, search_space=search_space, linq=linq),
            conv(num_features - d, num_features + d, kernel_size=3, k_bits=self.k_bits, bias=bias),
            nn.LeakyReLU(0.05)
        )
        self.compression = nn.Conv2d(num_features + d, num_features, kernel_size=1)
        self.loss_kdf= loss_kdf
       

    def forward(self, x):
        weighted_bits = x[4]
        f = x[3]
        bits = x[2]
        grad = x[0]
        x = x[1]

        residual = x

        # x = self.enhancement_top(x)
        flops=0

        grad,x,bits,weighted_bits= self.enhancement_top[0]([grad,x,bits,weighted_bits]) # BitSelector
        
        x = self.enhancement_top[1:3](x)
        grad,x,bits,weighted_bits = self.enhancement_top[3]([grad,x,bits,weighted_bits])
        
        x = self.enhancement_top[4:6](x)
        grad,x,bits,weighted_bits = self.enhancement_top[6]([grad,x,bits,weighted_bits])

        x = self.enhancement_top[7:9](x)


        slice_1 = x[:, :int((self.num_features - self.num_features/self.s)), :, :]
        slice_2 = x[:, int((self.num_features - self.num_features/self.s)):, :, :]

        grad,x,bits,weighted_bits = self.enhancement_bottom[0]([grad,slice_1,bits,weighted_bits])

        x = self.enhancement_bottom[1:3](x)
        grad,x,bits,weighted_bits = self.enhancement_bottom[3]([grad,x,bits,weighted_bits])

        x = self.enhancement_bottom[4:6](x)
        grad,x,bits,weighted_bits = self.enhancement_bottom[6]([grad,x,bits,weighted_bits])

        x = self.enhancement_bottom[7:9](x)
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

        # return x
        return [grad, x, bits, f, weighted_bits]


class IDN_CADyQ(nn.Module):
    def __init__(self, args, conv=quant_conv3x3, bias=False, k_bits=32):
        super(IDN_CADyQ, self).__init__()
        self.scale = args.scale[0]
        num_features = args.n_feats
        d = args.idn_d
        s = args.idn_s

        self.fblock = idn_pams.FBlock_PAMS(conv, num_features, bias=bias, k_bits=args.k_bits, ema_epoch=args.ema_epoch )
        
        m_dblocks = [
            DBlock_CADyQ(conv, num_features, d, s, bias=bias, k_bits=args.k_bits, ema_epoch=args.ema_epoch,search_space=args.search_space,loss_kdf=args.loss_kdf, linq=args.linq) for _ in range(args.n_resblocks)

        ]
        # args.n_resblocks should be 4
        self.dblocks = nn.Sequential(*m_dblocks)
        self.deconv = nn.ConvTranspose2d(num_features, 3, kernel_size=17, stride=self.scale, padding=8, output_padding=1)


    def forward(self, x):
        image = x
        bicubic = F.interpolate(x, scale_factor=self.scale, mode='bicubic', align_corners=False)

        grads: torch.Tensor = K.filters.spatial_gradient(K.color.rgb_to_grayscale(image/255.), order=1)
        grad = torch.mean(torch.abs(grads.squeeze(1)),(2,3)) *1e+3 # [16,2]

        x= self.fblock(x)

        f = None
        bits = 0; weighted_bits = 0

        grad, x,bits, f, weighted_bits= self.dblocks([grad, x, bits, f, weighted_bits])
        out = x

        x = self.deconv(x, output_size=bicubic.size())
        
        return bicubic + x, out, bits, f, weighted_bits
