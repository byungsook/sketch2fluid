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
from scipy import ndimage, misc
import platform
import os
import sys
sys.path.append('../')
# from utils import *
# from patch import *


# torch.manual_seed(42)
# np.random.seed(42)
# random.seed(42)

# timestep: 0.0416667
# currently leonhard cluster for hx ~720GB occupied
# data set: todo
# data range: dmax: 1.4354, dmin: 0.0000; vmax: 1.9781, vmin: -2.4203

class Loader(Dataset):
    def __init__(self, data_dir, window_size=10, steps=20):
        
        self.splitter = '/'
        if platform.system().lower() == 'windows':
            self.splitter = '\\'

        source_dir = data_dir + '/npz'
        sketch_dir = data_dir + '/sketch'
        self.source_dir = glob.glob(source_dir+'/*.npz')
        self.sketch_dir = glob.glob(sketch_dir+'/*.png')

        self.source_dir.sort(key=self.cmp_npz)
        self.sketch_dir.sort(key=self.cmp_sketch)
        
        # self.sketch_data = []
        # self.source_data = []

        # n_frames = len(sketch_dir) // 3
        
        # # for i in range(window_size-1,n_frames,window_size-1):
        # for i in range(0,n_frames,steps):
            
        #     self.sketch_data.append([sketch_dir[i*3+2],sketch_dir[i*3+0],sketch_dir[i*3+1]])
        #     self.source_data.append(source_dir[i])
        self.len = len(self.sketch_dir) // 3
        print ('===> Number of sketched keyframes:', self.len)
        # self.source_data.append(source_dir[0])
        # self.source_data.append(1) # dummy
                

    def cmp_npz(self, x):
        x = x.split(self.splitter)[-1].split('_')[-1].split('.')[0]
        x = int(x)
        # print (x)
        return x

    def cmp_sketch(self, x):
        s = x.split(self.splitter)[-1].split('_')
        x = s[0]
        v = s[-1].split('.')[0]
        x = int(x)
        # print (x)
        return x, v

    

    def load_data(self, index, plot=False):

        i = index
        # print (i)
        s_front = misc.imread( self.sketch_dir[i*3+2] ) / 255.0
        s_side = misc.imread( self.sketch_dir[i*3+0] ) / 255.0
        s_top = misc.imread( self.sketch_dir[i*3+1] ) / 255.0

        s_front = torch.FloatTensor(np.expand_dims(s_front, 0))
        s_side = torch.FloatTensor(np.expand_dims(s_side, 0))
        s_top = torch.FloatTensor(np.expand_dims(s_top, 0))
        ss = torch.cat([s_front, s_side, s_top], 0)

        f = np.load(self.source_dir[i])
        d = f['density']
        
        # res = d.shape[0]
        # if res < 129:
        #     p = 129 - res
        #     d = np.pad(d, ((0,p), (0, p), (0, p)), 'constant')
            # print (d.shape)
        d = torch.FloatTensor(np.expand_dims(d, 0))
        
        return ss, d

    def __getitem__(self, index):
        ss, d = self.load_data(index)
        return ss, d

        

    def __len__(self):
        return self.len

