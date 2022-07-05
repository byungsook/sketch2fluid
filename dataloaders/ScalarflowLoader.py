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

# torch.manual_seed(42)
# np.random.seed(42)
# random.seed(42)

class Loader(Dataset):
    # def __init__(self, data_dir, train_or_test, vmax, dmax=50, max_i):
    def __init__(self, data_dir, train_or_test, vmax=1, dmax=1, max_i=5, activ_d='tanh', window_size=4, ratio=1):

        self.test_list = ['000000', '000100']
        self.train_list = []
        for x in range(0,104):
            sim_idx = '{:0>6}'.format(x)
            if sim_idx not in self.test_list:
                self.train_list.append(sim_idx)

        print (len(self.test_list), len(self.train_list))
        
        self.start_frame = 31
        self.sim_list = glob.glob(data_dir+'/*')
        
        self.splitter = '/'
        if platform.system().lower() == 'windows':
            self.splitter = '\\'

        # self.den_list.sort(key=self.cmp)
        
        self.vmax = vmax
        self.dmax = dmax

        self.den_data = []
        self.vel_data = []
        
        for s in self.sim_list:
            sx = s.split(self.splitter)[-1].split('_')[-1]
            if train_or_test == 'test':
                if sx in self.test_list: 
                    self.den_list = glob.glob(s+'/*.npz')
                    self.den_list.sort(key=self.cmp)
                    for l in self.den_list:
                        l_tmp = l.split(self.splitter)[-1].split('_')
                        l_frame = int(l_tmp[-1].split('.')[0])
                        l_type = l_tmp[0]
                        # print (l_type, l_frame)
                        if l_frame >= self.start_frame:
                            if l_type == 'density':
                                self.den_data.append(l)
                            elif l_type == 'velocity':
                                self.vel_data.append(l)

            if train_or_test == 'train':
                if sx in self.train_list: 
                    self.den_list = glob.glob(s+'/*.npz')
                    self.den_list.sort(key=self.cmp)
                    for l in self.den_list:
                        l_tmp = l.split(self.splitter)[-1].split('_')
                        l_frame = int(l_tmp[-1].split('.')[0])
                        l_type = l_tmp[0]
                        # print (l_type, l_frame)
                        if l_frame >= self.start_frame:
                            if l_type == 'density':
                                self.den_data.append(l)
                            elif l_type == 'velocity':
                                self.vel_data.append(l)

        print (train_or_test+' samples:', len(self.den_data))
    
    def cmp(self, x):
        x = x.split(self.splitter)[-1].split('_')[-1]
        x = int(x.split('.')[0])
        return x

    def __getitem__(self, index):
        
        den_path = self.den_data[index]
        vel_path = self.vel_data[index]
        
        # print (data_path)
        # print (len(self.data))
        # print (data_path)
        d = np.load(den_path)['data'][...,0]
        # v = np.load(vel_path)['data'].transpose(3,0,1,2)
        
        bottom, top = 25, 24
        d = d[:,0+bottom:178-top:]
        d = np.pad(d, ((14,15), (0, 0), (14, 15)), 'constant')

        # bottom, top = 25, 25
        # d = d[:,0+bottom:178-top:]
        # d = np.pad(d, ((14,14), (0, 0), (14, 14)), 'constant')
        # cv = 15
        # d = d.clip(0, cv) / cv
        # 
        # d = d / self.dmax
        d = d / d.max()

        d = np.clip(d * 4, 0, 1) # newly added

        # print (d.shape, d.min(), d.max())
        # v = v[:,:,0+bottom:178-top:]
        # v = np.pad(v, ((0,0), (14,14), (0, 0), (14, 14)), 'constant')
        
        # v = v / self.vmax  # divided velocity max.
        # v = 2 * v - 1
        
        # print (d.shape, v.shape)
        # print (d.min(), d.max(), v.min(), v.max())

        # plt.figure(1)
        # plt.subplot(121)
        # plt.imshow(v[:,int(d.shape[0]//2)].transpose(1,2,0)*9.007, cmap='spring')

        # plt.subplot(122)
        # plt.imshow(d[int(d.shape[0]//2)], cmap=plt.cm.gnuplot2)    # 
        # plt.pause(0.1)
        # print (d.min(), d.max())
        d = torch.FloatTensor(d).unsqueeze(dim=0)
        # v = torch.FloatTensor(v)
        
        return d, den_path
        
    def __len__(self):
        return len(self.den_data)



# def main():
#     from torch.utils.data import DataLoader
#     from tqdm import tqdm
#     vmax, dmax = 9.007, 74.488
#     data = '../../DATA/scalarflow'
#     # train_dset = Loader(data, 'train', vmax, dmax)
#     # train_loader = DataLoader(train_dset, batch_size=1, shuffle=True, num_workers=0, drop_last=True, pin_memory=True)
#     test_dset = Loader(data, 'test', vmax, dmax)
#     test_loader = DataLoader(test_dset, batch_size=1, shuffle=False, num_workers=0, drop_last=True, pin_memory=True)
    
#     for i, (den, vel) in enumerate(tqdm(test_loader)):
#         pass

# if __name__ == '__main__':
#     main()


