import torch
from torch.utils.data import Dataset
import glob
import platform
import numpy as np
import random
import matplotlib.pyplot as plt
import time
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from scipy import ndimage, misc
import platform
import os
import sys
sys.path.append('../')

np.random.seed(42)
random.seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)

class Loader(Dataset):
    def __init__(self, data_dir, train_or_test, ratio=1):
        
        
        self.splitter = '/'
        if platform.system().lower() == 'windows':
            self.splitter = '\\'

        self.train_or_test = train_or_test
        
        # test_size = 4 # 4/665
        # self.data_nosource = self.get_dataset(data_dir+'/npz_regen', 1000, 996, 20, ratio) # 998
        # self.data_wsource = self.get_dataset(data_dir+'/npz_wsource_regen', 666, 662, 30, ratio)   # 664

        if self.train_or_test == 'train':
            self.data_nosource = self.get_dataset(data_dir+'/no_source/regen', 0, 985, 20, ratio) # 985, 1000
            self.data_wsource = self.get_dataset(data_dir+'/w_source/regen', 0, 656, 30, ratio)   # 656, 666
        elif self.train_or_test == 'test':
            self.data_nosource = self.get_dataset(data_dir+'/no_source/regen_test', 985, 1000, 20, ratio) # 985, 1000
            self.data_wsource = self.get_dataset(data_dir+'/w_source/regen_test', 656, 666, 30, ratio)   # 656, 666
        
        self.data = self.data_nosource + self.data_wsource 
        print ('loaded total '+self.train_or_test+' samples:', len(self.data))

    def get_dataset(self, data_dir, min_i, max_i, window_size, ratio=1):
        sim_list = glob.glob(data_dir+'/*.npz')
        # sim_list.sort(key=self.cmp)   # don't need to sort actually
        d_list = list(range(min_i, max_i))
        data = []
        for i in d_list:
            for j in range(0,window_size):
                f = '/out_{:0>5}.npz'.format(str(i*window_size+j))
                f = data_dir+f
                if os.path.exists(f):
                    data.append(f)
        return data

    def cmp(self, x):
        x = x.split(self.splitter)[-1].split('_')[-1]
        x = int(x.split('.')[0])
        return x


    def load_data(self, sim):
        data = np.load(sim)

        try:
            d = data['density']
        except:
            d = data['d'] # [0,1]
        # v = data['v'] / self.vmax # normalized to [-1,1]
        return d

    def __getitem__(self, index):
        ds = []
        fn = self.data[index]
        d0 = self.load_data(fn)
        ds.append(torch.FloatTensor(np.expand_dims(d0,0)))
        ds = torch.cat(ds, 0)
        
        return ds, fn
        

    def __len__(self):
        return len(self.data)

