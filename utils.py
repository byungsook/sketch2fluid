import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, grad

from math import ceil
import math
import numpy as np

import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
# from patch import *
from torch.nn import init
import patch_sketcher
import random

np.random.seed(42) # cpu vars
random.seed(42) # Python
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42) # gpu vars

def seq_sum(n):
    return n * (n + 1) // 2
    
def exists_or_create(dirname):
    """
    Check if directory exists.
    If yes returns true
    else create it and return false
    """
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        return False
    return True

def init_sphere_mask(device, res=129, r=64):
    # only once
    center = res // 2
    mask = np.ones((res,res,res) )
    for i in range(res):
        for j in range(res):
            for k in range(res):
                if (i-center)**2+(j-center)**2+(k-center)**2 > r**2:
                    mask[i,j,k] = 0

    mask = torch.FloatTensor(mask).unsqueeze(0).unsqueeze(0).to(device)
    return mask


def apply_sphere_mask(d, mask):
    return d * mask

def pad_for_rot(d, large_size):

    small_size = d.size(-1)
    diff = large_size - small_size
    if diff % 2 == 0:
        pad_size = [diff // 2, diff // 2] * 3
    else:
        pad_size = [diff // 2, diff // 2 + 1] * 3
    # print (large_size, math.ceil(small_size * math.sqrt(2)))
    
    # print (pad_size)
    d = F.pad(d, pad_size)
    # print (d.size())
    return d

def uniform_sampling(a, b):
    return a + np.random.random() * (b - a)

def gaussian_sampling(mu, sigma):
    return np.random.normal(mu, sigma)

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# def torch_equalize(image):
#     image = (image + 1) / 2
#     # print (image.size())
#     """Implements Equalize function from PIL using PyTorch ops based on:
#     https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py#L352"""
#     def scale_channel(im):
#         """Scale the data in the channel to implement equalize."""
#         # im = im[:, :, c]
#         im = im * 255
#         # print (im.min(), im.max())
#         # print (im.size())
#         # Compute the histogram of the image channel.
#         histo = torch.histc(im, bins=256, min=0, max=255)#.type(torch.int32)
#         # For the purposes of computing the step, filter out the nonzeros.
#         # print (histo.size())
#         nonzero_histo = torch.reshape(histo[histo != 0], [-1])
#         # print (nonzero_histo.size())
#         step = (torch.sum(nonzero_histo) - nonzero_histo[-1]) // 255
#         def build_lut(histo, step):
#             # Compute the cumulative sum, shifting by step // 2
#             # and then normalization by step.
#             lut = (torch.cumsum(histo, 0) + (step // 2)) // step
#             # Shift lut, prepending with 0.
#             lut = torch.cat([torch.zeros(1), lut[:-1]]) 
#             # Clip the counts to be in range.  This is done
#             # in the C code for image.point.
#             return torch.clamp(lut, 0, 255)

#         # If step is zero, return the original image.  Otherwise, build
#         # lut from the full histogram and step and then index from it.
#         if step == 0:
#             result = im
#         else:
#             # can't index using 2d index. Have to flatten and then reshape
#             result = torch.gather(build_lut(histo, step), 0, im.flatten().long())
#             result = result.reshape_as(im)
        
#         # return result.type(torch.uint8)
#         return result

#     # Assumes RGB for now.  Scales each channel independently
#     # and then stacks the result.
#     s1 = scale_channel(image[:,0,...]).unsqueeze(dim=1) / 255
#     # print (image.size(), s1.size())
#     print (s1.min(), s1.max(), image.min(), image.max())
#     import matplotlib.pyplot as plt
#     plt.figure(1)
#     plt.subplot(221)
#     plt.imshow(image[0,0,64]*255, cmap=plt.cm.gray)
#     plt.subplot(222)
#     plt.imshow(s1[0,0,64], cmap=plt.cm.gray)
#     plt.subplot(223)
#     # b, bins, patches = plt.hist(image[0,0,64].flatten()*255, 255)
#     # N, bins, patches = plt.hist(image[0,0].flatten(), bins=3, color="#777777")
#     _ = plt.hist(image[0,0].flatten(), bins=5)
#     plt.subplot(224)
#     b, bins, patches = plt.hist(s1[0,0].flatten(), bins=5, color="#777777")
#     # plt.xlim([0,255])
#     plt.show()
#     # s2 = scale_channel(image, 1)
#     # s3 = scale_channel(image, 2)
#     # image = torch.stack([s1, s2, s3], 2)
#     return image


# def semi_lagrange_advection(den, vel, dt, device, vmax):
#     '''
#     input: after denorm den = (den + 1) / 2
#     '''
#     # padding to get MAC Grid size
#     # print (den.size(), vel.size())
#     p3d = (0, 1, 0, 1, 0, 1, 0, 0)
#     # p3d = (1,0,1,0,1,0, 0, 0)
#     # print (vmax)
#     res = den.size(-1)
#     ratio = int(128 / (res-1))  # 129
#     vel = vel * vmax #/ ratio
#     # print (vel.min(), vel.max())

#     vel = F.pad(vel[0,0], p3d, "constant", 0)
#     # print (vel[0].size(), den[0].size())
#     # print (den.size(), vel.size())
#     # advection single-step
#     # if vel.size(0) == 1:
#     from diff_advect import advect_semi_lagrange

#     # print (vel[0].size())
#     advt = advect_semi_lagrange(vel, den[0,0], dt, device, order=1, rk_order=1)
#     advt = advt.unsqueeze(0).unsqueeze(0)
#     return advt


def sl_advection(den, vel, dt, vmax): # device
    '''
    input: after denorm den = (den + 1) / 2, non_residual multi-scale density
            dim: (2, 4, 1, 33, 33, 33)  # (bs, scale, c, z, y, x)
    '''

    from diff_advect_patch import advect_semi_lagrange
    vel = vel * vmax #/ 8
    # print (den.size(), vel.size())
    advt = advect_semi_lagrange(vel[:,0,...], den[:,0,...], dt, dx=1) # order=1
    advt = advt.unsqueeze(1)#.unsqueeze(1)
    # print (advt.size())
    return advt

def concat_velocities(vel_list, dt):

    """
    Concatenate a list of velocities in a single one.
    Not implemented for patch advection

    dim: (bs, 3, x, y, z)
    """

    # sum_lagrange (i in [0, n[) v_i = sum (i in [0, n[) v_i[x-dt*v i->n] 

    v = vel_list[-1]

    from diff_advect_patch import advect_semi_lagrange

    for i in range(len(vel_list)-2, 0, -1):
        v = v + advect_semi_lagrange(v, vel_list[i], dt)
    
    return v


def sl_advection_patch(den, vel, dt, vmax, scale=0): # device
    '''
    input: after denorm den = (den + 1) / 2, non_residual multi-scale density
            dim: (2, 4, 1, 33, 33, 33)  # (bs, scale, c, z, y, x)
    '''


    from diff_advect_patch import advect_patch

    
    den = torch_residuals(den)  # added whenever doing advection
    # print (den.size(), vel.size())  # [1, 4, 1, 33, 33, 33], [1, 4, 3, 33, 33, 33]
    # vel = torch_eval_patch(vel[:,scale:])
    # print (vel.min(), vel.max())
    # padding to get MAC Grid size
    # p3d = (0, 1, 0, 1, 0, 1, 0, 0)
    # # p3d = (1,0,1,0,1,0, 0, 0)
    # # print (vmax)
    res = den.size(-1)
    ratio = int(128 / (res-1))  # 129
    # print (ratio, vmax, vmax/ratio)
    vel = vel * vmax 
    # print (vel.size())
    # print (vel.min(), vel.max())

    # vel = F.pad(vel, p3d, "constant", 0)
    # # print (vel[0].size(), den[0].size())

    # # advection single-step
    # num_scale = vel.size(1)
    # # if vel.size(0) == 1:
    # advt_d = []
    # scale = 0
    # for scale in range(3,num_scale):
    # print (vel.size())
    # print (den.size())
    advt = advect_patch(vel, den, dt, scale) # order=1
    advt = advt.unsqueeze(1)#.unsqueeze(1)
    # advt_d.append(advt)
    # advt_d = torch.stack(advt_d,0).unsqueeze(0)
    # print (advt.size())
    return advt








def preprocess_npz(d, v_x, v_y, v_z):
    # print (v_x.shape)
    v_x = v_x.reshape(129,129,130)
    v_y = v_y.reshape(129,130,129)
    v_z = v_z.reshape(130,129,129)

    v_z = 0.5 * (v_z[:-1,...] + v_z[1:,...])
    # v_y = 0.5 * (v_y[:,:-1,...] + v_y[:,:-1,...])
    v_y = 0.5 * (v_y[:,:-1,...] + v_y[:,1:,...])
    v_x = 0.5 * (v_x[...,:-1] + v_x[...,1:])

    # v_x = 0.5 * (v_x[:-1,...] + v_x[1:,...])
    # # v_y = 0.5 * (v_y[:,:-1,...] + v_y[:,:-1,...])
    # v_y = 0.5 * (v_y[:,:-1,...] + v_y[:,1:,...])
    # v_z = 0.5 * (v_z[...,:-1] + v_z[...,1:])
    
    # v = np.stack((v_x, v_y, v_z), axis=-1)#[:,::-1,:,:].transpose(2,1,0,3)*128; v = v[:,::-1,:,:]
    v = np.stack((v_x, v_y, v_z), axis=-1)#[:,::-1,:,:].transpose(2,1,0,3)*128; v = v[:,::-1,:,:]
    
    # d = d.transpose(2,1,0)#[:,::-1,:]
    # d = d[:,::-1,:]
    # d = padGrid(d, 128)
    # v = v[:,::-1,:,:]
    # v = v.transpose(2,1,0,3)[:,::-1,:,:]#*129

    # print (d.shape, v.shape)
    # plt.figure(1)
    # plt.subplot(121)
    
    # plt.imshow(np.mean(d,0),cmap=plt.cm.gray)
    # plt.subplot(122)
    
    # plt.imshow(v[64,:,:,2])
    # plt.show()
    return d, v


    
def preprocess_vdb(d, v_x, v_y, v_z):

    '''
    input d: after d.transpose(2,1,0); d = padGrid(d, 128)
    input v: after v = np.stack((v_x, v_y, v_z), axis=-1)
    '''
    d = padDenGrid(d, 128)
    v_x = padVelGrid(v_x, (129,128,128))
    v_y = padVelGrid(v_y, (128,129,128))
    v_z = padVelGrid(v_z, (128,128,129))


    # v_x = np.pad(v_x, ((0,0), (0, 1), (0, 1)), 'constant')
    # v_y = np.pad(v_y, ((0,1), (0, 0), (0, 1)), 'constant')
    # v_z = np.pad(v_z, ((0,1), (0, 1), (0, 0)), 'constant')

    
    
    # v_x_tmp = v_x.copy()
    # v_x = v_x[1:,...]
    # v_y = v_y[:,1:,:]
    # v_z = v_z[...,1:]
    v_x = 0.5 * (v_x[:-1,...] + v_x[1:,...])
    # v_y = 0.5 * (v_y[:,:-1,...] + v_y[:,:-1,...])
    v_y = 0.5 * (v_y[:,:-1,...] + v_y[:,1:,...])
    v_z = 0.5 * (v_z[...,:-1] + v_z[...,1:])
    

    # print ('rehgioe')

    v_x = np.pad(v_x, ((0,1), (0, 1), (0, 1)), 'constant')
    v_y = np.pad(v_y, ((0,1), (0, 1), (0, 1)), 'constant')
    v_z = np.pad(v_z, ((0,1), (0, 1), (0, 1)), 'constant')
    d = np.pad(d, ((0,1), (0, 1), (0, 1)), 'constant')
    
    # print ( np.mean( (v_x_tmp[1:] - v_x)**2 ) )
    # plt.figure(1)
    # plt.subplot(121)
    # plt.imshow(v_x_tmp[64], cmap=plt.cm.RdBu)
    # plt.subplot(122)
    # plt.imshow(v_x[64], cmap=plt.cm.RdBu)
    # plt.show()

    # print (v_x.shape, v_y.shape, v_z.shape, d.shape)

    # print (d.shape, v_x.shape, v_y.shape, v_z.shape)
    # v_x = f['vel_x'].reshape(129,128,128)[1:,...]
    # v_y = f['vel_y'].reshape(128,129,128)[:,1:,:]
    # v_z = f['vel_z'].reshape(128,128,129)[...,1:]
    
    # v = np.stack((v_y, v_z, v_x), axis=0)[:,:,::-1,:]*20
    v = np.stack((v_x, v_y, v_z), axis=-1)#[:,::-1,:,:].transpose(2,1,0,3)*128; v = v[:,::-1,:,:]

    d = d.transpose(2,1,0)#[:,::-1,:]
    # d = padGrid(d, 128)

    v = v.transpose(2,1,0,3)*128


    return d, v

def padDenGrid(x, target=128):
    '''
    Pad grid with zeros after reading grid from .vdb files
    Coordinate: y-up, x-right, z-toeye
    '''
    # print (x.shape)
    dx = 128 - x.shape[0]
    dy = 128 - x.shape[1]
    dz = 128 - x.shape[2]
    x = np.pad(x,((0, dx), (0, dy), (0, dz)), 'constant')
    
    # print (x.shape)
    return x

def padVelGrid(v_, target=(129,128,128)):
    '''
    Pad grid with zeros after reading grid from .vdb files
    '''
    dx = target[0] - v_.shape[0]
    dy = target[1] - v_.shape[1]
    dz = target[2] - v_.shape[2]

    v_ = np.pad(v_, ((0, dx), (0, dy), (0, dz)), 'constant')
    # print (x.shape)
    return v_

def preprocess_resolution(d, max_res):
    if max_res == 3:    # 5->3 in patch-based
        return d
    # print (max_res)
    s = 4*int(pow(2, max_res))
    # d = F.adaptive_avg_pool3d(d, 64)
    # import matplotlib.pyplot as plt
    # d = torch.flip(d, [3])
    # print (d.size())
    d = F.interpolate(d, size=s)   # added for faster test
    # print (d.size())
    # plt.figure(1)
    # plt.imshow(d[0,0,32], cmap='gray')
    # plt.show()
    return d

def weights_init(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)


def hypersphere(z, radius=1):
#    return z * radius / z.norm(p=2, dim=1, keepdim=True)
    return z

def exp_mov_avg(Gs, G, alpha=0.999, global_step=999):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(Gs.parameters(), G.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


class Progress:
    """Determine the progress parameter of the training given the epoch and the progression in the epoch
    Args:
          n_iter (int): the number of epochs before changing the progress,
          pmax (int): the maximum progress of the training.
          batchSizeList (list): the list of the batchSize to adopt during the training
    """

    def __init__(self, n_iter, pmax, batchSizeList):
        assert n_iter > 0 and isinstance(n_iter, int), 'n_iter must be int >= 1'
        assert pmax >= 0 and isinstance(pmax, int), 'pmax must be int >= 0'
        assert isinstance(batchSizeList, list) and \
               all(isinstance(x, int) for x in batchSizeList) and \
               all(x > 0 for x in batchSizeList) and \
               len(batchSizeList) == pmax + 1, \
            'batchSizeList must be a list of int > 0 and of length pmax+1'

        self.n_iter = n_iter
        self.pmax = pmax
        self.p = 0
        self.batchSizeList = batchSizeList

    def progress(self, epoch, i, total):
        """Update the progress given the epoch and the iteration of the epoch
        Args:
            epoch (int): batch of images to resize
            i (int): iteration in the epoch
            total (int): total number of iterations in the epoch
        """
        x = (epoch + i / total) / self.n_iter
        self.p = min(max(int(x / 2), x - ceil(x / 2), 0), self.pmax)
        return self.p

    def resize(self, images):
        """Resize the images  w.r.t the current value of the progress.
        Args:
            images (Variable or Tensor): batch of images to resize
        """
        x = int(ceil(self.p))
        if x >= self.pmax:
            return images
        else:
            return F.adaptive_avg_pool3d(images, 4 * 2 ** x)

    @property
    def batchSize(self):
        """Returns the current batchSize w.r.t the current value of the progress"""
        x = int(ceil(self.p))
        return self.batchSizeList[x]


class GradientPenalty:
    """Computes the gradient penalty as defined in "Improved Training of Wasserstein GANs"
    (https://arxiv.org/abs/1704.00028)
    Args:
        batchSize (int): batch-size used in the training. Must be updated w.r.t the current batchsize
        lambdaGP (float): coefficient of the gradient penalty as defined in the article
        gamma (float): regularization term of the gradient penalty, augment to minimize "ghosts"
    """

    def __init__(self, batchSize, lambdaGP, gamma=1, device='cpu'):
        self.batchSize = batchSize
        self.lambdaGP = lambdaGP
        self.gamma = gamma
        self.device = device

    def __call__(self, netD, real_data, fake_data, progress):
        alpha = torch.rand(self.batchSize, 1, 1, 1, 1, requires_grad=True, device=self.device)
        # print (fake_data)
        # randomly mix real and fake data
        interpolates = real_data + alpha * (fake_data - real_data)
        # print (interpolates)
        # compute output of D for interpolated input
        disc_interpolates = netD(interpolates, progress)
        # print (disc_interpolates)
        # compute gradients w.r.t the interpolated outputs
        gradients = grad(outputs=disc_interpolates, inputs=interpolates,
                         grad_outputs=torch.ones(disc_interpolates.size(), device=self.device),
                         create_graph=True, retain_graph=True, only_inputs=True)[0].view(self.batchSize, -1)
        # gradients = gradients + 1e-16
        # gradient_penalty = (((gradients.norm(2, dim=1) - self.gamma) / self.gamma) ** 2).mean() * self.lambdaGP

        # print (gradients)
        gradient_penalty = torch.mean((1. - torch.sqrt(1e-8+torch.sum(gradients.view(gradients.size(0), -1)**2, dim=1)))**2) * self.lambdaGP
        # print (gradient_penalty)
        return gradient_penalty













def torchJacobian3(x, order=2):
    '''
    input x: (batch, channel, z, y, x) # y-down
    '''

    if order == 2:

        dudx = torch.cat((x[:,0,:,:,1:2] - x[:,0,:,:,0:1],
                            (x[:,0,:,:,2:] - x[:,0,:,:,:-2])/2,
                            x[:,0,:,:,-1:] - x[:,0,:,:,-2:-1]), dim=3)

        dudy = torch.cat((x[:,0,:,1:2] - x[:,0,:,0:1],
                            (x[:,0,:,2:] - x[:,0,:,:-2])/2,
                            x[:,0,:,-1:] - x[:,0,:,-2:-1]), dim=2)

        dudz = torch.cat((x[:,0,1:2] - x[:,0,0:1],
                            (x[:,0,2:] - x[:,0,:-2])/2,
                            x[:,0,-1:] - x[:,0,-2:-1]), dim=1)

        dvdx = torch.cat((x[:,1,:,:,1:2] - x[:,1,:,:,0:1],
                            (x[:,1,:,:,2:] - x[:,1,:,:,:-2])/2,
                            x[:,1,:,:,-1:] - x[:,1,:,:,-2:-1]), dim=3)

        dvdy = torch.cat((x[:,1,:,1:2] - x[:,1,:,0:1],
                            (x[:,1,:,2:] - x[:,1,:,:-2])/2,
                            x[:,1,:,-1:] - x[:,1,:,-2:-1]), dim=2)

        dvdz = torch.cat((x[:,1,1:2] - x[:,1,0:1],
                            (x[:,1,2:] - x[:,1,:-2])/2,
                            x[:,1,-1:] - x[:,1,-2:-1]), dim=1)

        dwdx = torch.cat((x[:,2,:,:,1:2] - x[:,2,:,:,0:1],
                            (x[:,2,:,:,2:] - x[:,2,:,:,:-2])/2,
                            x[:,2,:,:,-1:] - x[:,2,:,:,-2:-1]), dim=3)

        dwdy = torch.cat((x[:,2,:,1:2] - x[:,2,:,0:1],
                            (x[:,2,:,2:] - x[:,2,:,:-2])/2,
                            x[:,2,:,-1:] - x[:,2,:,-2:-1]), dim=2)

        dwdz = torch.cat((x[:,2,1:2] - x[:,2,0:1],
                            (x[:,2,2:] - x[:,2,:-2])/2,
                            x[:,2,-1:] - x[:,2,-2:-1]), dim=1)

    else:

        # x = torch.flip(x, [3])
        dudx = x[:,0,:,:,1:] - x[:,0,:,:,:-1]
        dvdx = x[:,1,:,:,1:] - x[:,1,:,:,:-1]
        dwdx = x[:,2,:,:,1:] - x[:,2,:,:,:-1]

        dudy = x[:,0,:,1:,:] - x[:,0,:,:-1,:] # no flipped y
        dvdy = x[:,1,:,1:,:] - x[:,1,:,:-1,:] # no flipped y
        dwdy = x[:,2,:,1:,:] - x[:,2,:,:-1,:] # no flipped y

        # dudy = x[:,0,:,:-1,:] - x[:,0,:,1:,:] # flipped y
        # dvdy = x[:,1,:,:-1,:] - x[:,1,:,1:,:] # flipped y
        # dwdy = x[:,2,:,:-1,:] - x[:,2,:,1:,:] # flipped y

        dudz = x[:,0,1:,:,:] - x[:,0,:-1,:,:]
        dvdz = x[:,1,1:,:,:] - x[:,1,:-1,:,:]
        dwdz = x[:,2,1:,:,:] - x[:,2,:-1,:,:]

        dudx = torch.cat((dudx, torch.unsqueeze(dudx[:,:,:,-1], dim=3)), dim=3)
        dvdx = torch.cat((dvdx, torch.unsqueeze(dvdx[:,:,:,-1], dim=3)), dim=3)
        dwdx = torch.cat((dwdx, torch.unsqueeze(dwdx[:,:,:,-1], dim=3)), dim=3)
        
        # dudy = torch.cat((torch.unsqueeze(dudy[:,:,0,:], dim=2), dudy), dim=2)    # flipped y
        # dvdy = torch.cat((torch.unsqueeze(dvdy[:,:,0,:], dim=2), dvdy), dim=2)    # flipped y
        # dwdy = torch.cat((torch.unsqueeze(dwdy[:,:,0,:], dim=2), dwdy), dim=2)    # flipped y
        
        
        dudy = torch.cat((dudy, torch.unsqueeze(dudy[:,:,-1,:], dim=2)), dim=2)   # no flipped y
        dvdy = torch.cat((dvdy, torch.unsqueeze(dvdy[:,:,-1,:], dim=2)), dim=2)   # no flipped y
        dwdy = torch.cat((dwdy, torch.unsqueeze(dwdy[:,:,-1,:], dim=2)), dim=2)   # no flipped y
        

        dudz = torch.cat((dudz, torch.unsqueeze(dudz[:,-1,:,:], dim=1)), dim=1)
        dvdz = torch.cat((dvdz, torch.unsqueeze(dvdz[:,-1,:,:], dim=1)), dim=1)
        dwdz = torch.cat((dwdz, torch.unsqueeze(dwdz[:,-1,:,:], dim=1)), dim=1)

    u = dwdy - dvdz
    v = dudz - dwdx
    w = dvdx - dudy
    
    j = torch.stack([dudx,dudy,dudz,dvdx,dvdy,dvdz,dwdx,dwdy,dwdz], dim=1)
    c = torch.stack([u,v,w], dim=1)

    # c = torch.flip(c, [3])
    # j = torch.flip(j, [3])
    
    # print (j.size(), c.size())
    return j, c
    


def torchGrad3D_den(x):
    x = x.squeeze(dim=2)
    # x: bczyx, y no flipped
    dudx = x[:,0,1:,:,:] - x[:,0,:-1,:,:]
    dudy = x[:,0,:,1:,:] - x[:,0,:,:-1,:]
    dudz = x[:,0,:,:,1:] - x[:,0,:,:,:-1]
    
    dudx = torch.cat((dudx, torch.unsqueeze(dudx[:,-1,:,:], dim=1)), dim=1)
    
    dudy = torch.cat((dudy, torch.unsqueeze(dudy[:,:,-1,:], dim=2)), dim=2)    # no flipped y
   
    dudz = torch.cat((dudz, torch.unsqueeze(dudz[:,:,:,-1], dim=3)), dim=3)
   
    j = torch.stack([dudx,dudy,dudz], dim=1)
    return j




def convert_png2mp4(imgdir, fps):
    
    import imageio
    
    filename = imgdir + '/a.mp4'
    # print (imgdir, filename)
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    try:
        writer = imageio.get_writer(filename, fps=fps)
    except Exception:
        imageio.plugins.ffmpeg.download()
        writer = imageio.get_writer(filename, fps=fps)

    imgs = sorted(glob.glob("{}/*.png".format(imgdir)))
    # imgs.sort(key=cmp)
    # print (imgs)

    for img in imgs:
        # print (img)
        fn = img.split('\\')[-1].split('_')
        # if int(fn[2])<109:
        #     continue
        # print (fn)

        # if fn[1] == '159' and fn[-2] == 'd' and fn[-1] == 'pred.png':
        #     print (fn)
        #     pass
        # else:
        #     continue
        im = imageio.imread(img)
        # print (im.shape, type(im))
        writer.append_data(im)
    writer.close()

# neural volume
# def xaviermultiplier(m, gain):
#     if isinstance(m, nn.Conv1d):
#         ksize = m.kernel_size[0]
#         n1 = m.in_channels
#         n2 = m.out_channels

#         std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
#     elif isinstance(m, nn.ConvTranspose1d):
#         ksize = m.kernel_size[0] // m.stride[0]
#         n1 = m.in_channels
#         n2 = m.out_channels

#         std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
#     elif isinstance(m, nn.Conv2d):
#         ksize = m.kernel_size[0] * m.kernel_size[1]
#         n1 = m.in_channels
#         n2 = m.out_channels

#         std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
#     elif isinstance(m, nn.ConvTranspose2d):
#         ksize = m.kernel_size[0] * m.kernel_size[1] // m.stride[0] // m.stride[1]
#         n1 = m.in_channels
#         n2 = m.out_channels

#         std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
#     elif isinstance(m, nn.Conv3d):
#         ksize = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2]
#         n1 = m.in_channels
#         n2 = m.out_channels

#         std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
#     elif isinstance(m, nn.ConvTranspose3d):
#         ksize = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] // m.stride[0] // m.stride[1] // m.stride[2]
#         n1 = m.in_channels
#         n2 = m.out_channels

#         std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
#     elif isinstance(m, nn.Linear):
#         n1 = m.in_features
#         n2 = m.out_features

#         std = gain * math.sqrt(2.0 / (n1 + n2))
#     else:
#         return None

#     return std

# def xavier_uniform_(m, gain):
#     std = xaviermultiplier(m, gain)
#     m.weight.data.uniform_(-std * math.sqrt(3.0), std * math.sqrt(3.0))

# def initmod(m, gain=1.0, weightinitfunc=xavier_uniform_):
#     validclasses = [nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d]
#     if any([isinstance(m, x) for x in validclasses]):
#         weightinitfunc(m, gain)
#         if hasattr(m, 'bias'):
#             m.bias.data.zero_()

#     # blockwise initialization for transposed convs
#     if isinstance(m, nn.ConvTranspose2d):
#         # hardcoded for stride=2 for now
#         m.weight.data[:, :, 0::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]
#         m.weight.data[:, :, 1::2, 0::2] = m.weight.data[:, :, 0::2, 0::2]
#         m.weight.data[:, :, 1::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]

#     if isinstance(m, nn.ConvTranspose3d):
#         # hardcoded for stride=2 for now
#         m.weight.data[:, :, 0::2, 0::2, 1::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
#         m.weight.data[:, :, 0::2, 1::2, 0::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
#         m.weight.data[:, :, 0::2, 1::2, 1::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
#         m.weight.data[:, :, 1::2, 0::2, 0::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
#         m.weight.data[:, :, 1::2, 0::2, 1::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
#         m.weight.data[:, :, 1::2, 1::2, 0::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
#         m.weight.data[:, :, 1::2, 1::2, 1::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]

# def initseq(s):
#     for a, b in zip(s[:-1], s[1:]):
#         if isinstance(b, nn.ReLU):
#             initmod(a, nn.init.calculate_gain('relu'))
#         elif isinstance(b, nn.LeakyReLU):
#             initmod(a, nn.init.calculate_gain('leaky_relu', b.negative_slope))
#         elif isinstance(b, nn.Sigmoid):
#             initmod(a)
#         elif isinstance(b, nn.Softplus):
#             initmod(a)
#         else:
#             initmod(a)

#     initmod(s[-1])














'''
init from pix2pix github
'''

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='xavier', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Return an initialized network.
    """
    # if len(gpu_ids) > 0:
    #     assert(torch.cuda.is_available())
    #     net.to(gpu_ids[0])
    #     net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net















def plot_eight_subfigures(x, filename):
    d0, gen_den, s_front_ss, s_front_recon, d0_rot, gen_den_rot, s_side_ss, s_side_recon = x

    plt.figure(1)
    plt.subplot(421)
    plt.imshow(torch.mean(d0.cpu()[0,0], 0), cmap=plt.cm.gnuplot2)  # , vmin=0, vmax=1
    plt.subplot(422)
    plt.imshow(torch.mean(gen_den.cpu()[0,0], 0), cmap=plt.cm.gnuplot2)
    plt.subplot(423)
    plt.imshow(s_front_ss.cpu()[0,0],cmap='gray', vmin=0, vmax=1)
    plt.subplot(424)
    plt.imshow(s_front_recon.cpu()[0,0],cmap='gray', vmin=0, vmax=1)
    plt.subplot(425)
    plt.imshow(torch.mean(d0_rot.cpu()[0,0], 0), cmap=plt.cm.gnuplot2)
    plt.subplot(426)
    plt.imshow(torch.mean(gen_den_rot.cpu()[0,0], 0), cmap=plt.cm.gnuplot2)
    plt.subplot(427)
    plt.imshow(s_side_ss.cpu()[0,0],cmap='gray', vmin=0, vmax=1)
    plt.subplot(428)
    plt.imshow(s_side_recon.cpu()[0,0],cmap='gray', vmin=0, vmax=1)
    plt.savefig(filename, bbox_indices='tight')
    plt.clf()




def plot_six_subfigures(x, filename):
    gen_den, s_front_recon, s_side_recon, s_front_ss, s_side_ss = x
    # print ('Before:', gen_den.min(), gen_den.max())
    # gen_den = gen_den / gen_den.max()
    # print ('After:', gen_den.min(), gen_den.max())
    plt.figure(1)
    plt.subplot(321)
    plt.imshow(torch.mean(gen_den.cpu()[0,0],0), cmap=plt.cm.gnuplot2)
    plt.subplot(322)
    plt.imshow(torch.mean(gen_den.cpu()[0,0].permute(2,1,0),0), cmap=plt.cm.gnuplot2)
    plt.subplot(323)
    plt.imshow(s_front_recon.cpu()[0,0], cmap=plt.cm.gray, vmin=0, vmax=1)
    plt.subplot(324)
    plt.imshow(s_side_recon.cpu()[0,0], cmap=plt.cm.gray, vmin=0, vmax=1)
    plt.subplot(325)
    plt.imshow(s_front_ss.cpu()[0,0], cmap=plt.cm.gray, vmin=0, vmax=1)
    plt.subplot(326)
    plt.imshow(s_side_ss.cpu()[0,0], cmap=plt.cm.gray, vmin=0, vmax=1)
    plt.savefig(filename, bbox_inches='tight')
    plt.clf()


def render_eight_subfigures(renderer, x, filename):
    gen_den, den, s_front_recon, s_side_recon, s_front_ss, s_side_ss = x

    gen_den_f = renderer(gen_den[0:1], lights=2, background='default')
    gen_den_f =  np.moveaxis(np.array(gen_den_f[0].cpu()), 0, 2)[::-1]
    gen_den_s = renderer(patch_sketcher.rotate(gen_den[0:1], 'xn'), lights=2, background='default')
    gen_den_s =  np.moveaxis(np.array(gen_den_s[0].cpu()), 0, 2)[::-1]

    den_f = renderer(den[0:1], lights=2, background='default')
    den_f =  np.moveaxis(np.array(den_f[0].cpu()), 0, 2)[::-1]
    den_s = renderer(patch_sketcher.rotate(den[0:1], 'xn'), lights=2, background='default')
    den_s =  np.moveaxis(np.array(den_s[0].cpu()), 0, 2)[::-1]
    
    plt.figure(1)
    plt.subplot(241)
    plt.imshow(gen_den_f)
    plt.subplot(242)
    plt.imshow(den_f)
    plt.subplot(243)
    plt.imshow(torch.flip(s_front_recon.cpu()[0,0], [0]), cmap=plt.cm.gray, vmin=0, vmax=1)
    plt.subplot(244)
    plt.imshow(torch.flip(s_front_ss.cpu()[0,0], [0]), cmap=plt.cm.gray, vmin=0, vmax=1)
    plt.subplot(245)
    plt.imshow(gen_den_s)
    plt.subplot(246)
    plt.imshow(den_s)
    plt.subplot(247)
    plt.imshow(torch.flip(s_side_recon.cpu()[0,0],[0]), cmap=plt.cm.gray, vmin=0, vmax=1)
    plt.subplot(248)
    plt.imshow(torch.flip(s_side_ss.cpu()[0,0],[0]), cmap=plt.cm.gray, vmin=0, vmax=1)
    plt.savefig(filename, bbox_inches='tight')
    plt.clf()


def render_four_subfigures(renderer, x, filename):
    gen_den, den = x

    gen_den_f = renderer(gen_den[0:1], lights=2, background='default')
    gen_den_f =  np.moveaxis(np.array(gen_den_f[0].cpu()), 0, 2)[::-1]
    gen_den_s = renderer(patch_sketcher.rotate(gen_den[0:1], 'xn'), lights=2, background='default')
    gen_den_s =  np.moveaxis(np.array(gen_den_s[0].cpu()), 0, 2)[::-1]

    den_f = renderer(den[0:1], lights=2, background='default')
    den_f =  np.moveaxis(np.array(den_f[0].cpu()), 0, 2)[::-1]
    den_s = renderer(patch_sketcher.rotate(den[0:1], 'xn'), lights=2, background='default')
    den_s =  np.moveaxis(np.array(den_s[0].cpu()), 0, 2)[::-1]
    
    plt.figure(1)
    plt.subplot(221)
    plt.imshow(gen_den_f)
    plt.subplot(222)
    plt.imshow(den_f)
    plt.subplot(223)
    plt.imshow(gen_den_s)
    plt.subplot(224)
    plt.imshow(den_s)
    plt.savefig(filename, bbox_inches='tight')
    plt.clf()



def plot_two_subfigures(x, filename):
    v1,v2 = x
    plt.figure(1)
    plt.subplot(121)
    plt.imshow(torch.mean(v1.cpu()[0,0],0))
    plt.subplot(122)
    plt.imshow(torch.mean(v2.cpu()[0,0],0))
    plt.savefig(filename, bbox_inches='tight')
    plt.clf()

def plot_histogram(x, filename, nbin=100):
    rem = x > 0.02
    x = x[rem].flatten()
    rem = x < 0.98
    x = x[rem].flatten()

    plt.figure(1)
    plt.hist(x.detach().cpu().numpy(), bins=nbin, normed=True)
    plt.savefig(filename, bbox_inches='tight')
    plt.clf()




















####################################################### save PIL #######################################################


def save_render(renderer, x, filename):
    f = renderer(x[0:1].detach(), background='default')
    if f.size(-1) < 258:
        f = F.interpolate(f, scale_factor=2, mode='bilinear')
    f =  np.moveaxis(np.array(f[0].cpu().numpy()), 0, 2)[::-1]
    Image.fromarray((f*255).astype(np.uint8)).save(filename)

def save_sketch(x, filename, out_size=258):
    x = F.interpolate(x.detach(), size=out_size, mode='bilinear')[0,0].cpu().numpy()[::-1]
    Image.fromarray((x*255).astype(np.uint8)).save(filename)





















def generate_perlin_noise_3d(shape, res):
    def f(t):
        return 6*t**5 - 15*t**4 + 10*t**3
    
    delta = (res[0] / shape[0], res[1] / shape[1], res[2] / shape[2])
    d = (shape[0] // res[0], shape[1] // res[1], shape[2] // res[2])
    grid = np.mgrid[0:res[0]:delta[0],0:res[1]:delta[1],0:res[2]:delta[2]]
    grid = grid.transpose(1, 2, 3, 0) % 1
    # Gradients
    theta = 2*np.pi*np.random.rand(res[0]+1, res[1]+1, res[2]+1)
    phi = 2*np.pi*np.random.rand(res[0]+1, res[1]+1, res[2]+1)
    gradients = np.stack((np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)), axis=3)
    g000 = gradients[0:-1,0:-1,0:-1].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g100 = gradients[1:  ,0:-1,0:-1].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g010 = gradients[0:-1,1:  ,0:-1].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g110 = gradients[1:  ,1:  ,0:-1].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g001 = gradients[0:-1,0:-1,1:  ].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g101 = gradients[1:  ,0:-1,1:  ].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g011 = gradients[0:-1,1:  ,1:  ].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g111 = gradients[1:  ,1:  ,1:  ].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    # Ramps
    n000 = np.sum(np.stack((grid[:,:,:,0]  , grid[:,:,:,1]  , grid[:,:,:,2]  ), axis=3) * g000, 3)
    n100 = np.sum(np.stack((grid[:,:,:,0]-1, grid[:,:,:,1]  , grid[:,:,:,2]  ), axis=3) * g100, 3)
    n010 = np.sum(np.stack((grid[:,:,:,0]  , grid[:,:,:,1]-1, grid[:,:,:,2]  ), axis=3) * g010, 3)
    n110 = np.sum(np.stack((grid[:,:,:,0]-1, grid[:,:,:,1]-1, grid[:,:,:,2]  ), axis=3) * g110, 3)
    n001 = np.sum(np.stack((grid[:,:,:,0]  , grid[:,:,:,1]  , grid[:,:,:,2]-1), axis=3) * g001, 3)
    n101 = np.sum(np.stack((grid[:,:,:,0]-1, grid[:,:,:,1]  , grid[:,:,:,2]-1), axis=3) * g101, 3)
    n011 = np.sum(np.stack((grid[:,:,:,0]  , grid[:,:,:,1]-1, grid[:,:,:,2]-1), axis=3) * g011, 3)
    n111 = np.sum(np.stack((grid[:,:,:,0]-1, grid[:,:,:,1]-1, grid[:,:,:,2]-1), axis=3) * g111, 3)
    # Interpolation
    t = f(grid)
    n00 = n000*(1-t[:,:,:,0]) + t[:,:,:,0]*n100
    n10 = n010*(1-t[:,:,:,0]) + t[:,:,:,0]*n110
    n01 = n001*(1-t[:,:,:,0]) + t[:,:,:,0]*n101
    n11 = n011*(1-t[:,:,:,0]) + t[:,:,:,0]*n111
    n0 = (1-t[:,:,:,1])*n00 + t[:,:,:,1]*n10
    n1 = (1-t[:,:,:,1])*n01 + t[:,:,:,1]*n11
    return ((1-t[:,:,:,2])*n0 + t[:,:,:,2]*n1)
    
def generate_fractal_noise_3d(shape, res, octaves=1, persistence=0.5):
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise_3d(shape, (frequency*res[0], frequency*res[1], frequency*res[2]))
        frequency *= 2
        amplitude *= persistence
    return noise
    

def generate_fractal_noise_3d_v2(shape, res, k=128):
    noise = np.zeros(shape)
    frequency = k // 128
    amplitude = 0.1

    noise += amplitude * generate_perlin_noise_3d(shape, (frequency*res[0], frequency*res[1], frequency*res[2]))

    # noise = generate_fractal_noise_3d((128, 128, 128), (4,4,4), 4)

    noise = np.pad(noise, ((1,0),(1,0),(1,0)), 'constant')
    noise = torch.FloatTensor(noise).unsqueeze(0).unsqueeze(0).repeat(1,3,1,1,1)
    
    return noise







