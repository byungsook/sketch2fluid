import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torchvision.models

import patch
import patch_sketcher
import matplotlib.pyplot as plt
import scipy

class LSM(nn.Module):
    '''
        Least-square method for density initialization based on two-view sketches
    '''
    def __init__(self, device, preprocess='hole'):
        super(LSM, self).__init__()
        
        self.device = device
        self.preprocess = preprocess
        self.gaussian_smoother = patch_sketcher.GaussianSmoothing(1, 1, 1, 3).to(self.device)
        
        self.smoothing_iterations = 2

    def _preprocess(self, s):
        # under-shoot
        s[s<1] = 1 - s[s<1]
        s[s>=1] = 0
        # plt.figure(1)
        # plt.imshow(s[0,0].detach().cpu(),cmap='gray', vmin=0, vmax=1)
        # plt.colorbar()
        # plt.show()
        return s

    def _preprocess1(self, s):
        # under-shoot
        s[s>=0.8]=1
        s=1-s
        
        # binary errosion
        # s_cpu = s.detach().cpu().numpy()[0,0]
        # s_cpu = scipy.ndimage.binary_erosion(s_cpu).astype(s_cpu.dtype)
        # s_cpu = torch.FloatTensor(s_cpu).unsqueeze(0).unsqueeze(0)

        # plt.figure(1)
        # plt.imshow(1-s[0,0].detach().cpu(),cmap='gray', vmin=0, vmax=1)
        # plt.colorbar()
        # plt.show()
        return s

    def _preprocess2(self, s):
        s[s>=0.951] = 0 
        return s

    def _preprocess3(self, s):
        # binary
        s[s>0.96] = 1 # 0.96
        s[s<1] = 0
        s = 1 - s
        s = s * 0.3092 * 2
        return s

    def _preprocess4(self, s):
        s[s<=0.3]=1
        s = 1 - s
        # s[s<1]=1-s[s<1]
        # s[s>=0.5]=0
        return s

    def _extrude(self, x, preprocess):
        s = x.clone()
        if preprocess == 'hole':
            s = self._preprocess1(s)
        if preprocess == 'binary':
            s = self._preprocess3(s)
        
        rdim = s.size(-1)
        d = s.repeat(1,rdim,1,1)

        # d = d / rdim
        return d.unsqueeze(1)
    

    def lsm_3view(self, ss, lsm_scale):
        s1, s2, s3 = ss
        s1 = F.interpolate(s1, scale_factor=0.5, mode='bilinear')
        s2 = F.interpolate(s2, scale_factor=0.5, mode='bilinear')
        s3 = F.interpolate(s3, scale_factor=0.5, mode='bilinear')
        
        # print (s1.size())
        d1 = self._extrude(s1)
        d2 = self._extrude(s2)
        d3 = self._extrude(s3)
        
        d2 = patch_sketcher.rotate_back(d2, 'xn')
        d3 = patch_sketcher.rotate_back(d3, 'yp')
        
        d = d1 * d2 * d3
        
        # d = torch.sqrt(d + 1e-7)
        # print (d.max())
        d = d / d.max()
        
        d = d * lsm_scale
        d = torch.clamp(d, 0, 1)
        return d

    def lsm_2view(self, ss, lsm_scale, preprocess):
        s1, s2 = ss
        # s1 = patch.torch_downsample(s1)
        # s2 = patch.torch_downsample(s2)
        s1 = F.interpolate(s1, scale_factor=0.5, mode='bilinear')
        s2 = F.interpolate(s2, scale_factor=0.5, mode='bilinear')
        # print (s1.size())
        d1 = self._extrude(s1, preprocess)
        d2 = self._extrude(s2, preprocess)
        d2 = patch_sketcher.rotate_back(d2, 'xn')
        d = d1 * d2
        
        # d = torch.sqrt(d + 1e-7)
        # print (d.max())
        # d = d / d.max()
        
        d = d * lsm_scale
        d = torch.clamp(d, 0, 1)
        return d


    def lsm_1view(self, ss, preprocess):
        s1, = ss
        # s1[s1>0.4]=1
        # s1 = patch.torch_downsample(s1)
        s1 = F.interpolate(s1, scale_factor=0.5, mode='bilinear')
        d1 = self._extrude(s1, preprocess)

        # d1 = d1 / d1.max()
        # print (d1.size(), d1.min(), d1.max())
        return d1

    def forward(self, ss, lsm_scale=1):
        lsm_view = len(ss)
        
        if lsm_view == 1:
            return self.lsm_1view(ss, self.preprocess)
        elif lsm_view == 2:
            d = self.lsm_2view(ss, lsm_scale, 'binary')
            for i in range(self.smoothing_iterations):
                d2 = self.lsm_2view(ss, lsm_scale, self.preprocess)
                d = torch.clamp(d + d2, 0, 1)
                d = self.gaussian_smoother.to(d.device)(d)
            # d = self.lsm_2view(ss, lsm_scale, self.preprocess)
            return d
        elif lsm_view == 3:
            return self.lsm_3view(ss, lsm_scale, self.preprocess)









