import torch
import torch.nn as nn
import torch.nn.functional as F

from model import carn
from model.quant_ops import quant_act_pams, quant_conv3x3

class BasicBlock_PAMS(nn.Module):
    def __init__(self,
                 in_channels, out_channels, conv, k_bits=32, bias=False, ema_epoch=1,
                 ksize=3, stride=1, pad=1):
        super(BasicBlock_PAMS, self).__init__()

        self.quant_act = quant_act_pams(k_bits, ema_epoch)


        self.body = nn.Sequential(
            self.quant_act,
            conv(in_channels, out_channels, k_bits=k_bits, bias=bias, kernel_size=ksize, stride=stride, padding=pad),
            nn.ReLU(inplace=True)
        )

        carn.init_weights(self.modules)
        
    def forward(self, x):
        out = self.body(x)
        return out

class ResidualBlock_PAMS(nn.Module):
    def __init__(self, 
                 in_channels, out_channels,  conv, k_bits=32, bias=False, ema_epoch=1,loss_kdf= False, linq=False):
        super(ResidualBlock_PAMS, self).__init__()

        self.quant_act1 = quant_act_pams(k_bits,ema_epoch=ema_epoch)
        self.quant_act2 = quant_act_pams(k_bits,ema_epoch=ema_epoch)

        self.body = nn.Sequential(
            self.quant_act1,
            conv(in_channels, out_channels, k_bits=k_bits, bias=bias, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            self.quant_act2,
            conv(out_channels, out_channels, k_bits=k_bits, bias=bias, kernel_size=3, stride=1, padding=1),
        )
        self.loss_kdf= loss_kdf

        carn.init_weights(self.modules)
        
    def forward(self, x):
        f = x[1]
        x = x[0]

        out = self.body(x)

        f1 = out
        out = F.relu(out + x)

        if self.loss_kdf:
            if f is None:
                f = f1.unsqueeze(0)
            else: 
                f = torch.cat([f, f1.unsqueeze(0)], dim=0)
        else:
            f = None


        # return out
        return [out, f]

class CARNBlock_PAMS(nn.Module):
    def __init__(self, 
                 in_channels, out_channels, conv, k_bits=32, bias=False, ema_epoch=1, loss_kdf=False, group=1,  linq=False, fully=False):
        super(CARNBlock_PAMS, self).__init__()

        self.b1 = ResidualBlock_PAMS(64, 64, conv, k_bits=k_bits, bias=bias, ema_epoch=ema_epoch,loss_kdf=loss_kdf,linq=linq)
        self.b2 = ResidualBlock_PAMS(64, 64, conv, k_bits=k_bits, bias=bias, ema_epoch=ema_epoch,loss_kdf=loss_kdf,linq=linq)
        self.b3 = ResidualBlock_PAMS(64, 64, conv, k_bits=k_bits, bias=bias, ema_epoch=ema_epoch,loss_kdf=loss_kdf,linq=linq)

        if fully:
            self.c1 = BasicBlock_PAMS(64*2, 64, conv, k_bits=k_bits, bias=bias,ema_epoch=ema_epoch, ksize=1, stride=1, pad=0)
            self.c2 = BasicBlock_PAMS(64*3, 64, conv, k_bits=k_bits, bias=bias,ema_epoch=ema_epoch, ksize=1, stride=1, pad=0)
            self.c3 = BasicBlock_PAMS(64*4, 64, conv, k_bits=k_bits, bias=bias,ema_epoch=ema_epoch, ksize=1, stride=1, pad=0)
        else:
            self.c1 = carn.BasicBlock(64*2, 64, 1, 1, 0)
            self.c2 = carn.BasicBlock(64*3, 64, 1, 1, 0)
            self.c3 = carn.BasicBlock(64*4, 64, 1, 1, 0)

    def forward(self, x):
        f = x[1]
        x = x[0]

        c0 = o0 = x

        b1,f = self.b1([o0,f])
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)
        
        b2,f = self.b2([o1,f])
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)
        
        b3,f = self.b3([o2,f])
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        return [o3, f]


class CARN_PAMS(nn.Module):
    def __init__(self, args, conv=quant_conv3x3, bias=False, k_bits=32, multi_scale=False, linq=False, fully=False):
        super(CARN_PAMS, self).__init__()
        
        scale = args.scale[0]
        # multi_scale = args.multi_scale
        # multi_scale = False
        group = args.group 
        self.fully = fully

        self.k_bits = args.k_bits

        self.sub_mean = carn.MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = carn.MeanShift((0.4488, 0.4371, 0.4040), sub=False)
        
        if self.fully:
            self.entry = conv(3, 64, k_bits=args.k_bits, bias=bias, kernel_size=3, stride=1, padding=1)
            self.quant_head = quant_act_pams(k_bits=args.k_bits, ema_epoch=args.ema_epoch)

        else:
            self.entry = nn.Conv2d(3, 64, 3, 1, 1)

        self.b1 = CARNBlock_PAMS(64, 64, conv, k_bits=args.k_bits, bias=bias,ema_epoch=args.ema_epoch,loss_kdf=args.loss_kdf,linq=linq,fully=fully)
        self.b2 = CARNBlock_PAMS(64, 64, conv, k_bits=args.k_bits, bias=bias,ema_epoch=args.ema_epoch,loss_kdf=args.loss_kdf,linq=linq,fully=fully)
        self.b3 = CARNBlock_PAMS(64, 64, conv, k_bits=args.k_bits, bias=bias,ema_epoch=args.ema_epoch,loss_kdf=args.loss_kdf,linq=linq,fully=fully)

        if self.fully:
            self.c1 = BasicBlock_PAMS(64*2, 64, conv, k_bits=args.k_bits, bias=bias,ema_epoch=args.ema_epoch, ksize=1, stride=1, pad=0)
            self.c2 = BasicBlock_PAMS(64*3, 64, conv, k_bits=args.k_bits, bias=bias,ema_epoch=args.ema_epoch, ksize=1, stride=1, pad=0)
            self.c3 = BasicBlock_PAMS(64*4, 64, conv, k_bits=args.k_bits, bias=bias,ema_epoch=args.ema_epoch, ksize=1, stride=1, pad=0)
        else:
            self.c1 = carn.BasicBlock(64*2, 64, 1, 1, 0)
            self.c2 = carn.BasicBlock(64*3, 64, 1, 1, 0)
            self.c3 = carn.BasicBlock(64*4, 64, 1, 1, 0)

        self.upsample = carn.UpsampleBlock(64, scale=scale, multi_scale=multi_scale, group=group, fully=fully, k_bits=args.k_bits)

        if self.fully:
            self.exit = conv(64, 3, k_bits=args.k_bits, bias=bias, kernel_size=3, stride=1, padding=1)
            # self.quant_tail = quant_act_lin(k_bits=args.k_bits)
            self.quant_tail = quant_act_pams(k_bits=args.k_bits, ema_epoch=args.ema_epoch)
        else:
            self.exit = nn.Conv2d(64, 3, 3, 1, 1)
                
    def forward(self, x, scale):
    # def forward(self, x, scale=4):
        f = None
        x = self.sub_mean(x)
        if self.fully: x = self.quant_head(x)

        x = self.entry(x)
        c0 = o0 = x

        b1,f = self.b1([o0,f])
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)
        
        b2,f = self.b2([o1,f])
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)
        
        b3,f = self.b3([o2,f])
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        feat = o3
        
        out = self.upsample(o3, scale=scale)

        if self.fully: out = self.quant_tail(out)
        out = self.exit(out)
        out = self.add_mean(out)

        return out, feat, f