import torch
from torch.utils.data import Dataset
import glob
import platform
import numpy as np
import matplotlib.pyplot as plt
import time
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import random
from scipy import ndimage, misc, stats
import platform
import os
from PIL import Image

import sys
sys.path.append('../')
from utils import *
from patch import *


# torch.manual_seed(42)
# np.random.seed(42)
# random.seed(42)

# timestep: 0.0416667
# currently leonhard cluster for hx ~720GB occupied
# data set: todo
# data range: dmax: 1.4354, dmin: 0.0000; vmax: 1.9781, vmin: -2.4203

class Loader(Dataset):
    def __init__(self, data_dir, window_size=10, sigma=0.8):
        
        self.splitter = '/'
        if platform.system().lower() == 'windows':
            self.splitter = '\\'
        sketch_dir = data_dir
        self.sketch_data = []
        self.source_data = []
        self.sigma = sigma
        s_front, s_side, s_top = [], [], []
        # sketch_dir0 = glob.glob(sketch_dir+'/front_compressed/*')   # front
        # sketch_dir1 = glob.glob(sketch_dir+'/side_compressed/*')   # Side
        # sketch_dir2 = glob.glob(sketch_dir+'/top_compressed/*')    # Top

        sketch_dir0 = glob.glob(sketch_dir+'/front/*')   # front; front_compressed
        sketch_dir1 = glob.glob(sketch_dir+'/Side/*')   # Side; Side_compressed
        sketch_dir2 = glob.glob(sketch_dir+'/Top/*')    # Top; Top_compressed

        sketch_dir0.sort(key=self.cmp_sketch)
        sketch_dir1.sort(key=self.cmp_sketch)
        sketch_dir2.sort(key=self.cmp_sketch)
        
        n_frames = len(sketch_dir0)
        # print (n_frames)
        for s in sketch_dir0:
            # print (s)
            s_front.append(s)
        for s in sketch_dir1:
            # print (s)
            s_side.append(s)
        for s in sketch_dir2:
            # print (s)
            s_top.append(s)

        for i in range(n_frames):
            # self.sketch_data.append([s_front[i], s_side[i], s_top[i]])
            # print ( s_front[i], s_side[i], s_top[i] )
            # print (platform.system().lower())
            # if platform.system().lower() == 'darwin':
            self.sketch_data.append([s_front[i], s_side[i], s_top[i]])
            # elif platform.system().lower() == 'linux':
            #     self.sketch_data.append([s_top[i], s_front[i], s_side[i]])
                
        self.source_data = np.zeros((128,128,128))

        center = 128
        radius = 128
        self.circle_mask = np.ones((258,258))
        for i in range(0,258):
            for j in range(0,258):
                if (i-center)**2+(j-center)**2 > radius**2:
                    self.circle_mask[i,j] = 0

        rad = 12
        src_height = 12
        src_z_pos, src_y_pos, src_x_pos = 64, 12, 64
        for z in range(src_z_pos-rad, src_z_pos + rad):
            for y in range(src_y_pos-rad, src_y_pos+rad):
                for x in range(src_x_pos-rad, src_x_pos+rad):
                    if (z - src_z_pos)**2 + (y - src_y_pos)**2 + (x - src_x_pos)**2 - rad**2 < 1e-6: #and x > 0 and z > 0:
                        self.source_data[z,y,x] = 1



    def cmp_npz(self, x):
        x = x.split(self.splitter)[-1].split('_')[-1].split('.')[0]
        x = int(x)
        # print (x)
        return x

    def cmp_sketch(self, x):
        
        s = x.split(self.splitter)[-1].split('.')[0]
        s = s[-4:]
        s = int(s)
        return s

    
    def _preprocess_single_sketch(self, s, sigma=0.1, res=257, interp='lanczos'):
        # s = ndimage.gaussian_filter(s, sigma=sigma)
        # s = ndimage.gaussian_filter(s, sigma=sigma)
        s = misc.imresize(s, (res,res), interp=interp, mode=None) / 255.0
        # print (s.min(), s.max())
        # print (sigma)
        s = ndimage.gaussian_filter(s, sigma=sigma) ####################### TBD
        # s = 1 - s
        # s = np.pad(s, ((29,29),(29,29)), 'constant')
        # s = 1 - s
        # s = 1 - s
        # s = s * self.circle_mask
        # s = 1 - s
        s = torch.FloatTensor(np.expand_dims(s, 0))
        return s

    def load_image_to_numpy(self, path):
        im = Image.open(path).convert('L')
        im = np.asarray(im)
        return im

    def load_data(self, index):
        # print (index)
        # print (len(self.sketch_data))
        sketches = []

        for i in range(0,len(self.sketch_data)):

            # s_front = misc.imread( self.sketch_data[i][0] ) #/ 255.0
            # s_side = misc.imread( self.sketch_data[i][1] ) # / 255.0
            # s_top = misc.imread( self.sketch_data[i][2] ) # / 255.0


            s_front = self.load_image_to_numpy(self.sketch_data[i][0])
            s_side = self.load_image_to_numpy(self.sketch_data[i][1])
            s_top = self.load_image_to_numpy(self.sketch_data[i][2])
            
            
            # crop
            start = 0
            end = 650
            # s_front = s_front[80:180,30:220]
            # s_side = s_side[80:180,30:220]
            # s_top = s_top[80:180,30:220]
            
            # print (s_front.shape, s_front)
            # better crop
            # s_front = s_front[150:750,100:650,:]
            # s_side = s_side[150:750,100:570,:]
            # s_top = s_top[200:680,150:600,:]
            
            # print (s_front.shape, s_front.min(), s_front.max())

            res = 258 # 129, 257, 384
            s_front = self._preprocess_single_sketch(s_front, res=res, sigma=self.sigma)
            s_side = self._preprocess_single_sketch(s_side, res=res, sigma=self.sigma)
            s_top = self._preprocess_single_sketch(s_top, res=res, sigma=self.sigma)
            # print (s_front.shape)
            
            ss = torch.cat([s_front, s_side, s_top], 0)

            sketches.append(ss)

            # plt.figure(1)
            # plt.subplot(131)
            # plt.imshow(ss[0], cmap=plt.cm.gray, vmin=0, vmax=1)
            # plt.subplot(132)
            # plt.imshow(ss[1], cmap=plt.cm.gray, vmin=0, vmax=1)
            # plt.subplot(133)
            # plt.imshow(ss[2], cmap=plt.cm.gray, vmin=0, vmax=1)
            # plt.pause(2)

        sketches = torch.stack(sketches,0)
        
        
        d = self.source_data
        res = d.shape[0]
        if res < 129:
            p = 129 - res
            d = np.pad(d, ((0,p), (0, p), (0, p)), 'constant')
        # print (sketches.size())
        d = torch.FloatTensor(np.expand_dims(d, 0))
        # plt.figure(1)
        # plt.subplot(331)
        # plt.imshow(torch.mean(torch.flip(d[0],[1]),0),cmap=plt.cm.gnuplot2)
        # plt.subplot(332)
        # plt.imshow(sketches[0,0],cmap=plt.cm.gray)
        # plt.subplot(333)
        # plt.imshow(sketches[0,0]+10*torch.mean(torch.flip(d[0],[1]),0),cmap=plt.cm.gray)
        # plt.subplot(334)
        # plt.imshow(torch.mean(torch.flip(d[0],[1]).permute(2,1,0),0),cmap=plt.cm.gnuplot2)
        # plt.subplot(335)
        # plt.imshow(sketches[0,1],cmap=plt.cm.gray)
        # plt.subplot(336)
        # plt.imshow(sketches[0,1]+10*torch.mean(torch.flip(d[0],[1]).permute(2,1,0),0),cmap=plt.cm.gray)

        # plt.subplot(337)
        # plt.imshow(torch.mean(torch.flip(d[0],[1]).permute(1,0,2),0),cmap=plt.cm.gnuplot2)
        # plt.subplot(338)
        # plt.imshow(sketches[0,2],cmap=plt.cm.gray)
        # plt.subplot(339)
        # plt.imshow(sketches[0,2]+10*torch.mean(torch.flip(d[0],[1]).permute(1,0,2),0),cmap=plt.cm.gray)

        # plt.show()
        return sketches, d

    def __getitem__(self, index):
        
        sketches, d = self.load_data(index)
        
        # print (sketches.size(), d.size())
        return sketches, d

        

    def __len__(self):
        return 1

