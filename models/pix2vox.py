from math import ceil
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import *
from utils import *
from patch import *


import torch
import torchvision.models


class Pix2VoxPair_sm(nn.Module):
    def __init__(self, bn=False, zdim=256, droprate=0.5, nout=3, activ=nn.LeakyReLU(0.2, True)):
        super(Pix2VoxPair_sm, self).__init__()
        
        self.activ = activ
        self.bn = bn
        self.nout = nout
        self.droprate = droprate

        if nout == 1:
            nin = 1
            # self.dec_4 = conv3d(64, nout, activ=nn.Tanh(), ks=4, s=1, p=2, bn=False)
        if nout == 3:
            nin = 1
            # self.dec_4 = conv3d(64, nout, activ=None, ks=4, s=1, p=2, bn=False)

        self.enc_1 = conv2d(nin, 64, activ=self.activ, ks=4, s=2, p=1, bn=False)
        self.enc_2 = conv2d(64+129, 256, activ=self.activ, ks=4, s=2, p=1, bn=self.bn) # p=1
        self.enc_3 = conv2d(256, 256, activ=self.activ, ks=4, s=2, p=1, bn=self.bn) # p=1
        self.enc_4 = conv2d(256, 512, activ=self.activ, ks=4, s=2, p=1, bn=self.bn) # 160
        self.enc_5 = conv2d(512, 512, activ=self.activ, ks=4, s=2, p=1, bn=self.bn)
        self.enc_6 = conv2d(512, 512, activ=self.activ, ks=4, s=2, p=1, bn=self.bn)
        self.enc_7 = conv2d(512, 512, activ=self.activ, ks=4, s=2, p=1, bn=self.bn)
        self.enc_8 = conv2d(512, 512, activ=self.activ, ks=4, s=2, p=1, bn=False)
        
        self.dec_8 = conv2d_up(512, 512, activ=self.activ, ks=3, s=1, p=1, bn=self.bn, dropout=None)
        self.dec_7 = conv2d_up(1024, 512, activ=self.activ, ks=3, s=1, p=1, bn=self.bn, dropout=None)
        self.dec_6 = conv2d_up(1024, 512, activ=self.activ, ks=3, s=1, p=1, bn=self.bn, dropout=None)
        self.dec_5 = conv2d_up(1024, 512, activ=self.activ, ks=3, s=1, p=1, bn=self.bn, dropout=None)
        self.dec_4 = conv2d_up(1024, 256, activ=self.activ, ks=3, s=1, p=1, bn=self.bn, dropout=None)
        self.dec_3 = conv2d_up(512, 256, activ=self.activ, ks=3, s=1, p=1, bn=self.bn, dropout=None)     # 128   
        self.dec_2 = conv2d_up(256, 256, activ=self.activ, ks=3, s=1, p=1, bn=self.bn, dropout=None)
        self.dec_1 = conv2d(256, 129, activ=None, ks=2, s=1, p=1, bn=False)
        
    '''
    258
        129                                                                        128 -> 129
            64------------------------------------------------------------------64
                32--------------------------------------------------------32
                    16--------------------------------------16
                        8-----------------------------8
                            4---------------4
                                2---------2
                                    1
    '''

    def forward(self, gen_den, sketch):

        in_den = gen_den

        sketch = 2 * sketch - 1
        gen_den = gen_den.squeeze(1)
        gen_den = 2 * gen_den - 1

        # print (sketch.size(), gen_den.size())
        e1 = self.enc_1(sketch)
        # print (e1.size())
        e1 = torch.cat([e1, gen_den], 1)
        # print (e1.size())
        e2 = self.enc_2(e1)
        # print (e2.size())
        e3 = self.enc_3(e2)
        # print (e3.size())
        e4 = self.enc_4(e3)
        # print (e4.size())
        e5 = self.enc_5(e4)
        # print (e5.size())
        e6 = self.enc_6(e5)
        # print (e6.size())
        e7 = self.enc_7(e6)
        # print (e7.size())
        e8 = self.enc_8(e7)
        # print (e8.size())

        d8 = self.dec_8(e8)
        # print (d8.size())
        d7 = self.dec_7(torch.cat([d8, e7], 1)) # d7 = self.dec_7(e7)
        # print (d7.size())
        d6 = self.dec_6(torch.cat([d7, e6], 1)) # d6 = self.dec_6(e6)
        # print (d6.size())
        d5 = self.dec_5(torch.cat([d6, e5], 1))
        # print (d5.size())
        d4 = self.dec_4(torch.cat([d5, e4], 1))
        # print (d4.size())
        d3 = self.dec_3(torch.cat([d4, e3], 1))
        # print (d3.size())
        # d2 = self.dec_2(torch.cat([d3, e2], 1)) 
        d2 = self.dec_2(d3) 
        
        # print (d2.size())
        d1 = self.dec_1(d2).unsqueeze(1)
        # d1 = d2.unsqueeze(1)
        # print (d1.size())
        
        y = in_den + d1
        y = torch.clamp(y, 0, 1)
        
        return y, d1



















class Pix2VoxPairVel(nn.Module):
    def __init__(self, bn=False, zdim=256, droprate=0.5, nout=3, activ=nn.LeakyReLU(0.2, True)):
        super(Pix2VoxPairVel, self).__init__()
        
        self.activ = activ
        self.bn = bn
        self.nout = nout
        self.droprate = droprate

        if nout == 1:
            nin = 1
            # self.dec_4 = conv3d(64, nout, activ=nn.Tanh(), ks=4, s=1, p=2, bn=False)
        if nout == 3:
            nin = 1
            # self.dec_4 = conv3d(64, nout, activ=None, ks=4, s=1, p=2, bn=False)

        self.enc_1 = conv2d(nin, 64, activ=self.activ, ks=4, s=2, p=1, bn=False)
        self.enc_2 = conv2d(64+129, 256, activ=self.activ, ks=4, s=2, p=1, bn=self.bn) # p=1
        self.enc_3 = conv2d(256, 256, activ=self.activ, ks=4, s=2, p=1, bn=self.bn) # p=1
        self.enc_4 = conv2d(256, 512, activ=self.activ, ks=4, s=2, p=1, bn=self.bn) # 160
        self.enc_5 = conv2d(512, 512, activ=self.activ, ks=4, s=2, p=1, bn=self.bn)
        self.enc_6 = conv2d(512, 512, activ=self.activ, ks=4, s=2, p=1, bn=self.bn)
        self.enc_7 = conv2d(512, 512, activ=self.activ, ks=4, s=2, p=1, bn=self.bn)
        self.enc_8 = conv2d(512, 512, activ=self.activ, ks=4, s=2, p=1, bn=False)
        
        self.dec_8 = conv2d_up(512, 512, activ=self.activ, ks=3, s=1, p=1, bn=self.bn, dropout=None)
        self.dec_7 = conv2d_up(1024, 512, activ=self.activ, ks=3, s=1, p=1, bn=self.bn, dropout=None)
        self.dec_6 = conv2d_up(1024, 512, activ=self.activ, ks=3, s=1, p=1, bn=self.bn, dropout=None)
        self.dec_5 = conv2d_up(1024, 512, activ=self.activ, ks=3, s=1, p=1, bn=self.bn, dropout=None)
        self.dec_4 = conv2d_up(1024, 256, activ=self.activ, ks=3, s=1, p=1, bn=self.bn, dropout=None)
        self.dec_3 = conv2d_up(512, 256, activ=self.activ, ks=3, s=1, p=1, bn=self.bn, dropout=None)     # 128   
        self.dec_2 = conv2d_up(256, 256, activ=self.activ, ks=3, s=1, p=1, bn=self.bn, dropout=None)
        
        self.dec_c1 = conv2d(256, 129, activ=None, ks=2, s=1, p=1, bn=False)
        self.dec_c2 = conv2d(256, 129, activ=None, ks=2, s=1, p=1, bn=False)
        self.dec_c3 = conv2d(256, 129, activ=None, ks=2, s=1, p=1, bn=False)
         
    '''
    258
        129                                                                        128 -> 129
            64------------------------------------------------------------------64
                32--------------------------------------------------------32
                    16--------------------------------------16
                        8-----------------------------8
                            4---------------4
                                2---------2
                                    1
    '''

    def forward(self, gen_den, sketch):

        in_den = gen_den

        sketch = 2 * sketch - 1
        gen_den = gen_den.squeeze(1)
        gen_den = 2 * gen_den - 1

        # print (sketch.size(), gen_den.size())
        e1 = self.enc_1(sketch)
        # print (e1.size())
        e1 = torch.cat([e1, gen_den], 1)
        # print (e1.size())
        e2 = self.enc_2(e1)
        # print (e2.size())
        e3 = self.enc_3(e2)
        # print (e3.size())
        e4 = self.enc_4(e3)
        # print (e4.size())
        e5 = self.enc_5(e4)
        # print (e5.size())
        e6 = self.enc_6(e5)
        # print (e6.size())
        e7 = self.enc_7(e6)
        # print (e7.size())
        e8 = self.enc_8(e7)
        # print (e8.size())

        d8 = self.dec_8(e8)
        # print (d8.size())
        d7 = self.dec_7(torch.cat([d8, e7], 1))
        # d7 = self.dec_7(e7)
        # print (d7.size())
        d6 = self.dec_6(torch.cat([d7, e6], 1))
        # d6 = self.dec_6(e6)
        
        # print (d6.size())
        d5 = self.dec_5(torch.cat([d6, e5], 1))
        # print (d5.size(), e4.size())
        d4 = self.dec_4(torch.cat([d5, e4], 1))
        # print (d4.size())
        d3 = self.dec_3(torch.cat([d4, e3], 1))
        # print (d3.size())
        # d2 = self.dec_2(torch.cat([d3, e2], 1)) 
        d2 = self.dec_2(d3)
        
        # print (d2.size())
        c1 = self.dec_c1(d2).unsqueeze(1)
        c2 = self.dec_c2(d2).unsqueeze(1)
        c3 = self.dec_c3(d2).unsqueeze(1)
        
        y = torch.cat([c1,c2,c3],1)

        # print (y.size())
        
        return y





class Pix2VoxPairOptim(nn.Module):
    def __init__(self, bn=False, zdim=256, droprate=0.5, nout=3, activ=nn.LeakyReLU(0.2, True)):
        super(Pix2VoxPairOptim, self).__init__()
        
        self.w_residual = torch.nn.Parameter(torch.zeros(1,1,129,129,129))
        self.w_residual.requires_grad = True

        self.w_d = torch.nn.Parameter(torch.rand(1,1,129,129,129))
        # print (self.w_d.min(), self.w_d.max())
        
        self.w_d.requires_grad = True

        self.w_v = torch.nn.Parameter(torch.zeros(1,3,129,129,129)+0.01)
        # self.w_v = torch.nn.Parameter(-0.01+0.02*torch.rand(1,3,129,129,129))
        # print (self.w_v.min(), self.w_v.max())
        
        self.w_v.requires_grad = True
        
        # print (self.w.size())
        self.use_curl = False

    def forward(self, gen_den, sketch, mode):
        
        if mode == 'residual':
            y = gen_den + self.w_residual
            y = torch.clamp(y, 0, 1)
        elif mode == 'direct':
            y = torch.clamp(self.w_d, 0, 1)
        elif mode == 'velocity':
            y = self.w_v
            if self.use_curl:
                _, y = torchJacobian3(y)

        return y






class OptimParams(nn.Module):
    def __init__(self, bn=False, zdim=256, droprate=0.5, nout=3, activ=nn.LeakyReLU(0.2, True)):
        super(OptimParams, self).__init__()
        
        self.w_residual = torch.nn.Parameter(torch.zeros(1,1,129,129,129) + 0.5)
        self.w_residual.requires_grad = True

        self.w_d = torch.nn.Parameter(torch.rand(1,1,129,129,129))
        # print (self.w_d.min(), self.w_d.max())
        
        self.w_d.requires_grad = True

        self.w_v = torch.nn.Parameter(torch.zeros(1,3,129,129,129))
        # self.w_v = torch.nn.Parameter(-0.01+0.02*torch.rand(1,3,129,129,129))
        # print (self.w_v.min(), self.w_v.max())
        
        self.w_v.requires_grad = True
        
        # print (self.w.size())
        self.use_curl = False

    def forward(self, gen_den, mode):
        
        if mode == 'residual':
            y = gen_den + self.w_residual
            y = torch.clamp(y, 0, 1)
        elif mode == 'direct':
            y = torch.clamp(self.w_d, 0, 1)
        elif mode == 'velocity':
            y = self.w_v
            if self.use_curl:
                _, y = torchJacobian3(y)

        return y


