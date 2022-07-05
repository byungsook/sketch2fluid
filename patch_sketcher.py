#%%
# lastest version

import math
import numbers
import torch
import glob
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
np.random.seed(42)

import patch

import utils


##### Utility functions #####

def rotate(d, dr):

    

    return {
        "zp": lambda: d, 
        "zn": lambda: torch.flip(torch.flip(d, [2]), [4]),
        "xp": lambda: torch.flip(d.transpose(2, 4), [4]), 
        "xn": lambda: torch.flip(d.transpose(2, 4),[2]), 
        "yp": lambda: torch.flip(d.transpose(2, 3), [3]), 
        "yn": lambda: torch.flip(d.transpose(2, 3), [2]),


        "zp1": lambda: torch.flip(torch.flip(d, [3]), [4]), 
        "zp2": lambda: torch.flip(d.transpose(3,4), [3]),
        "zp3": lambda: torch.flip(torch.flip(torch.flip(d, [3]), [4]).transpose(3,4),[3]),
        
        "zn1": lambda: torch.flip(torch.flip(torch.flip(torch.flip(d, [2]), [4]), [3]), [4]), 
        "zn2": lambda: torch.flip(torch.flip(torch.flip(d, [2]), [4]).transpose(3,4), [3]),
        "zn3": lambda: torch.flip(torch.flip(torch.flip(torch.flip(torch.flip(d, [2]), [4]), [3]), [4]).transpose(3,4),[3]),
        
        "xp1": lambda: torch.flip(torch.flip(torch.flip(d.transpose(2, 4), [4]), [3]), [4]), 
        "xp2": lambda: torch.flip(torch.flip(d.transpose(2, 4), [4]).transpose(3,4), [3]),
        "xp3": lambda: torch.flip(torch.flip(torch.flip(torch.flip(d.transpose(2, 4), [4]), [3]), [4]).transpose(3,4),[3]),

        "xn1": lambda: torch.flip(torch.flip(torch.flip(d.transpose(2, 4),[2]), [3]), [4]), 
        "xn2": lambda: torch.flip(torch.flip(d.transpose(2, 4),[2]).transpose(3,4), [3]),
        "xn3": lambda: torch.flip(torch.flip(torch.flip(torch.flip(d.transpose(2, 4),[2]), [3]), [4]).transpose(3,4),[3]),
        
        "yp1": lambda: torch.flip(torch.flip(torch.flip(d.transpose(2, 3), [3]), [3]), [4]), 
        "yp2": lambda: torch.flip(torch.flip(d.transpose(2, 3), [3]).transpose(3,4), [3]),
        "yp3": lambda: torch.flip(torch.flip(torch.flip(torch.flip(d.transpose(2, 3), [3]), [3]), [4]).transpose(3,4),[3]),
        
        "yn1": lambda: torch.flip(torch.flip(torch.flip(d.transpose(2, 3), [2]), [3]), [4]), 
        "yn2": lambda: torch.flip(torch.flip(d.transpose(2, 3), [2]).transpose(3,4), [3]),
        "yn3": lambda: torch.flip(torch.flip(torch.flip(torch.flip(d.transpose(2, 3), [2]), [3]), [4]).transpose(3,4),[3]),
        

    }[dr]()


def rotate_back(d, dr):
    return {
        "zp": lambda: d,
        "zn": lambda: torch.flip(torch.flip(d,[4]),[2]),
        "xp": lambda: torch.flip(d,[4]).transpose(2, 4),
        "xn": lambda: torch.flip(d,[2]).transpose(2, 4),
        "yp": lambda: torch.flip(d,[3]).transpose(2, 3),
        "yn": lambda: torch.flip(d,[2]).transpose(2, 3),
        
        "zp1": lambda: torch.flip(torch.flip(d,[4]),[3]),
        "zp2": lambda: torch.flip(d,[3]).transpose(3,4),
        "zp3": lambda: torch.flip(torch.flip(torch.flip(d, [3]).transpose(3,4),[4]),[3]),
        
        "zn1": lambda: torch.flip(torch.flip(torch.flip(torch.flip(d,[4]),[3]),[4]),[2]),
        "zn2": lambda: torch.flip(torch.flip(torch.flip(d,[3]).transpose(3,4),[4]),[2]),
        "zn3": lambda: torch.flip(torch.flip(torch.flip(torch.flip(torch.flip(d, [3]).transpose(3,4),[4]),[3]),[4]),[2]),
        
        "xp1": lambda: torch.flip(torch.flip(torch.flip(d,[4]),[3]),[4]).transpose(2, 4),
        "xp2": lambda: torch.flip(torch.flip(d,[3]).transpose(3,4),[4]).transpose(2, 4),
        "xp3": lambda: torch.flip(torch.flip(torch.flip(torch.flip(d, [3]).transpose(3,4),[4]),[3]),[4]).transpose(2, 4),
        
        "xn1": lambda: torch.flip(torch.flip(torch.flip(d,[4]),[3]),[2]).transpose(2, 4),
        "xn2": lambda: torch.flip(torch.flip(d,[3]).transpose(3,4),[2]).transpose(2, 4),
        "xn3": lambda: torch.flip(torch.flip(torch.flip(torch.flip(d, [3]).transpose(3,4),[4]),[3]),[2]).transpose(2, 4),
        
        "yp1": lambda: torch.flip(torch.flip(torch.flip(d,[4]),[3]),[3]).transpose(2, 3),
        "yp2": lambda: torch.flip(torch.flip(d,[3]).transpose(3,4),[3]).transpose(2, 3),
        "yp3": lambda: torch.flip(torch.flip(torch.flip(torch.flip(d, [3]).transpose(3,4),[4]),[3]),[3]).transpose(2, 3),
        
        "yn1": lambda: torch.flip(torch.flip(torch.flip(d,[4]),[3]),[2]).transpose(2, 3),
        "yn2": lambda: torch.flip(torch.flip(d,[3]).transpose(3,4),[2]).transpose(2, 3),
        "yn3": lambda: torch.flip(torch.flip(torch.flip(torch.flip(d, [3]).transpose(3,4),[4]),[3]),[2]).transpose(2, 3),
        

    }[dr]()

def rotate_grad(g, dr):
    rg = rotate(g, dr)

    return {
        "zp": lambda: torch.stack([rg[:, 0], rg[:, 1], rg[:, 2]], dim=1), 
        "zn": lambda: torch.stack([-rg[:, 0], rg[:, 1], -rg[:, 2]], dim=1),
        "xp": lambda: torch.stack([-rg[:, 2], rg[:, 1], rg[:, 0]], dim=1), 
        "xn": lambda: torch.stack([rg[:, 2], rg[:, 1], -rg[:, 0]], dim=1), 
        "yp": lambda: torch.stack([ rg[:, 0], -rg[:, 2], rg[:, 1]], dim=1), 
        "yn": lambda: torch.stack([ rg[:, 0], rg[:, 2], -rg[:, 1]], dim=1)
    }[dr]()


def rotate_patch_pos(pos, dr, coarse_size):
    """128
    coarse_size: triplet, size of the coarse volume
    """

    #full volume size (indice of last element)
    #last = (np.array(self.coarse_volume.size()[2:])-1)*2
    last = (np.array(coarse_size)-1)
    return np.array({
        "zp": lambda: pos, 
        "zn": lambda: (last[0] - pos[0], pos[1], last[2] - pos[2]),
        "xp": lambda: (pos[2], pos[1], last[0] - pos[0]), 
        "xn": lambda: (last[2] - pos[2], pos[1], pos[0]), 
        "yp": lambda: (pos[1],  last[0] - pos[0], pos[2]),
        "yn": lambda: (last[1] - pos[1], pos[0], pos[2])
    }[dr]())


def inverse_rotate_patch_pos(pos, dr, coarse_size):
    """128
    coarse_size: triplet, size of the coarse volume
    """

    #full volume size (indice of last element)
    #last = (np.array(self.coarse_volume.size()[2:])-1)*2
    last = (np.array(coarse_size)-1)
    return np.array({
        "zp": lambda: pos, 
        "zn": lambda: (last[0] - pos[0], pos[1], last[2] - pos[2]),
        "xp": lambda: (last[2] - pos[2], pos[1], pos[0]), 
        "xn": lambda: (pos[2], pos[1], last[0] - pos[0]), 
        "yp": lambda: (last[1] - pos[1],  pos[0], pos[2]),
        "yn": lambda: (pos[1], last[0] - pos[0], pos[2])
    }[dr]())

def torchGrad3D(x, order = 2):

    if order == 1:
        dudx = torch.cat((x[:,0,:,:,0:1],
                            x[:,0,:,:,1:] - x[:,0,:,:,:-1]), dim=3)

        dudy = torch.cat((x[:,0,:,0:1],
                            x[:,0,:,1:] - x[:,0,:,:-1]), dim=2)

        dudz = torch.cat((x[:,0,0:1],
                            x[:,0,1:] - x[:,0,:-1]), dim=1)
        
        return torch.stack([dudx,dudy,dudz], dim=1)

    elif order == 2:

        dudx = torch.cat((x[:,0,:,:,1:2] - x[:,0,:,:,0:1],
                            (x[:,0,:,:,2:] - x[:,0,:,:,:-2])/2,
                            x[:,0,:,:,-1:] - x[:,0,:,:,-2:-1]), dim=3)

        dudy = torch.cat((x[:,0,:,1:2] - x[:,0,:,0:1],
                            (x[:,0,:,2:] - x[:,0,:,:-2])/2,
                            x[:,0,:,-1:] - x[:,0,:,-2:-1]), dim=2)

        dudz = torch.cat((x[:,0,1:2] - x[:,0,0:1],
                            (x[:,0,2:] - x[:,0,:-2])/2,
                            x[:,0,-1:] - x[:,0,-2:-1]), dim=1)
        
        return torch.stack([dudx,dudy,dudz], dim=1)

    else:
        raise ValueError("Order {} not supported".format(order))


    


def integrate(t, v, x = None):

    if x is not None:
        dx = (x[:, :, 1:, :, :] - x[:, :, :-1, :, :]) + 1e-7
        return ( # front bondary is very important to prevent the mask for disapearing for high densities at the front boundary
            (1-t[:, :, 0, :, :]) / (x[:, :, 0, :, :]+1e-7)*v[:, :, 0, :, :] + #front boundary, assume value of x is 0 #and v is unchanged
            #t[:, :, -1, :, :]*v[:, :, -1, :, :] + #back boundary ! removed for easier background extraction !
            torch.sum( (t[:, :, :-1, :, :] - t[:, :, 1:, :, :])/dx * (v[:, :, :-1, :, :] + v[:, :, 1:, :, :]) *.5, dim = 2))
    else:
        return ( # front bondary is very important to prevent the mask for disapearing for high densities at the front boundary
            (1-t[:, :, 0, :, :])*v[:, :, 0, :, :] + #front boundary, assume value of x is 0 #and v is unchanged
            #t[:, :, -1, :, :]*v[:, :, -1, :, :] + #back boundary ! removed for easier background extraction !
            torch.sum( (t[:, :, :-1, :, :] - t[:, :, 1:, :, :]) * (v[:, :, :-1, :, :] + v[:, :, 1:, :, :]) *.5, dim = 2))


def sketch_weights(d, dx, c):

    d = torch.flip(d, [2])
        
    x = torch.cumsum(d*dx, dim = 2)
    t = (x * c + 1) * torch.exp(-c*x)
    
    integrated =  torch.cat((
        t[:, :, 0:1, :, :] - t[:, :, 1:2, :, :],
        (t[:, :, :-2, :, :] - t[:, :, 2:, :, :])*.5,
        t[:, :, -2:-1 :, :] - t[:, :, -1:, :, :],
    ), dim = 2)

    return torch.flip(integrated, [2])

def inverse_sketch_loss(d, dx, c, d_prev, alpha):

    w = torch.clamp(1-alpha*sketch_weights(d, dx, c), min=0, max=1)
    return torch.mean(w*(d-d_prev))



################################################       Gaussian smoothing     ################################################


class GaussianSmoothing(nn.Module):
    """
        Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """


    def __init__(self, channels=1, kernel_half_size=3, sigma=1.6, dim=3):

        super(GaussianSmoothing, self).__init__()

        kernel_size = 1+2*kernel_half_size
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )

        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                       torch.exp((-((mgrid - mean) / std) ** 2) / 2)
                       
        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.conv_block = nn.Sequential()

        self.conv = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=kernel_half_size)
        self.conv.weight.data.copy_(kernel)
        self.conv.bias.data.copy_(torch.from_numpy(np.array([0.0])))

        self.conv_block.add_module("conv3d", self.conv)

    def forward(self, x):

        for layer in self.conv_block:
            x = layer(x)
        return x


################################################ Diff Patch Sketcher functions ################################################

class DiffPatchSketchRender(nn.Module):
    """


        dirs is the list of view direction, among ("xp", "xn", "yp", "yp", "zp", "zn").
        default is None, will be convertted to "zp". "xp" means the camera is placed on the positive x axis and look toward the negative x direction
        screen axis are then:
        xp up y, right -z
        xn up y, right +z
        yp up -z, right x
        yn up z, right x
        zp up y, right x
        zn up y, right -x

        coarse_volume is in [0,1]
        patch_sketch_info is a dict {dir(string): info}

    """
    def __init__(self, 
                 contour_thresh=0.8, kernel_half_size=3, sigma=1.6/1,     # 1.6
                 dirs=None, 
                 dx=1, patch_sketch_info = None, upsample_to = None):

        super(DiffPatchSketchRender, self).__init__()
        
        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # added

        self.contour_thresh = contour_thresh
        self.gaussian_smoother = GaussianSmoothing(1, kernel_half_size, sigma/dx, 3)

        self.dirs = dirs
        if self.dirs is None:
            self.dirs = ['zp']

        self.dx = dx
        self.patch_sketch_info = patch_sketch_info
        self.upsample_to= upsample_to
        

    def forward(self, x, toon_color=0.8, contour_thresh=0.8, light_dir=[0.5774,0.5774,0.5774]):      
        #d = self.gaussian_smoother(patch.torch_upsample((x+ 1) / 2))
        d = (x + 1) / 2
        d = self.gaussian_smoother(d)
        # output -grad to have a vector field that leaves the smoke 
        grad = -torchGrad3D(d) / self.dx

        out = []
        for dr in self.dirs:
            out = out + [self.surface_render(rotate(d, dr), rotate_grad(grad, dr), toon_color, contour_thresh, light_dir, None if self.patch_sketch_info is None else self.patch_sketch_info[dr] )]
            
        # returns a list of tuples (sketch, toon, shade, combined)

        return out

        
    def normal(self, d, grad, c=5, patch_sketch_info = None):

        if patch_sketch_info is None:
            x0, grad0, grad1, grad2, a0, a1, a2 = (0,)*7
        else:
            x0, grad0, grad1, grad2, a0, a1, a2 = patch_sketch_info

        # camera looks toward -z so we need to invert the z axsis

        d = torch.flip(d, [2])
        grad = torch.flip(grad, [2])

        if x0 is not 0:
            x0 = x0.unsqueeze(2) 
        x = x0 + torch.cumsum(d*self.dx, dim = 2)
        x_f = x[:,:, -1]

        t = (x * c + 1) * torch.exp(-c*x)
        coef = torch.exp(-c*x_f)
        grads = grad0 + integrate(t, grad, x) + coef * (grad2 + c*x_f*grad1)
        
        # normalize the avaeraged gradient to turn it into a normal
        #print (grads.size(), torch.sqrt(torch.sum(grads**2, dim=1)+1e-7).size())
        igrad = grads / torch.sqrt(torch.sum(grads**2, dim=1)+1e-7).unsqueeze(1)    # added unsqueeze for batch_size>1

        # compute mask
        a = torch.clamp(a0 
                        + integrate(t, torch.ones_like(d, requires_grad=False))
                        + coef * (a2 + c*x_f*a1), 0, 1)

        return a, igrad


    def torchToon(self, s, c=0.1):
        def smoothclamp(x, mi, mx): 
            return (lambda t: torch.where(t < 0 , torch.zeros_like(t, requires_grad=False), torch.where( t <= 1 , 3*t**2-2*t**3, torch.ones_like(t, requires_grad=False) ) ) )( (x-mi)/(mx-mi) )
        return (smoothclamp(s, .5-c, .5+c))


    def surface_render(self, d, grad, toon_color, contour_thresh, light_dir, patch_sketch_info=None):
        '''
        render sketch from a zp
        '''
        extinct_scale = 5
        toon_hardness = 0.1
        smoke_color = 0.95 # 0.95

        # added
        if toon_color >= smoke_color:
            toon_color = smoke_color - 0.01
        if contour_thresh >= 1:
            contour_thresh = 0.99
        
        #backgroud normal for sketch extraction. Aligned with view direction
        bgs = grad.new_tensor([0,0,1]).reshape((1,3,1,1))
        # bgs2 = grad.new_tensor([1,1,1]).reshape((1,3,1,1))

        #light direction for shading/toonshading
        l = grad.new_tensor(light_dir).reshape((1,3,1,1)) # [0.5774,-0.5774,0.5774]

        if self.upsample_to is not None:
            d = torch.nn.functional.interpolate(d, size = (d.shape[2], self.upsample_to, self.upsample_to), mode = 'trilinear', align_corners=False)
            grad = torch.nn.functional.interpolate(grad, size = (d.shape[2], self.upsample_to, self.upsample_to), mode = 'trilinear', align_corners=False)

        # reuse gradient and normals
        a, igrad = self.normal(d, grad, c=extinct_scale, patch_sketch_info = patch_sketch_info) # c=5
        
        nm = (1 - a) * bgs + a * igrad                  # [b, 3, 224, 224]

        s = torch.clamp(nm[:, 2:3], 0, contour_thresh) / contour_thresh     # 0.6, self.contour_thresh
        

        ######## s = nm[:, 2:3] # differentiable
        # s[s>contour_thresh] = contour_thresh
        # s[s<0] = 0
        # s = s / contour_thresh

        nm2 = (1 - a) * l + a * igrad
        # print (a.shape, l.shape, nm2.shape)
        
        sh = torch.clamp(torch.sum(l * nm2, dim=1), 0, 1).unsqueeze(dim=1)
        
        # toon shading
        t = self.torchToon(sh, c=toon_hardness)    # [b, 1, 224, 224], c=0.1                  
        # t = 1 - a + a * (toon_color + (1 - toon_color) * t)        # 0.75 + 0.25 * t
        t = (1 - a) + a * (toon_color + (smoke_color - toon_color) * t)
        # t = (toon_color + (smoke_color - toon_color) * t)
        # print (a.shape, l.shape, t.shape)
        # blending
        out = (1 - s) * s + s * t


        out = out * 2 - 1
        s   = s * 2 - 1
        t   = t * 2 - 1
        sd   = sh * 2 - 1
        
        # return s, t, sd, (1 - a) * bgs2 + a * igrad , out
        return s, t, sd, out

class DiffVorticeSketchRender(nn.Module):
    """


        dirs is the list of view direction, among ("xp", "xn", "yp", "yp", "zp", "zn").
        default is None, will be convertted to "zp". "xp" means the camera is placed on the positive x axis and look toward the negative x direction
        screen axis are then:
        xp up y, right -z
        xn up y, right +z
        yp up -z, right x
        yn up z, right x
        zp up y, right x
        zn up y, right -x


    """
    def __init__(self, kernel_half_size=3, sigma=1.6/1,
                 dirs=None, 
                 dx=1):

        super().__init__()
        
        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # added

        self.dirs = dirs
        if self.dirs is None:
            self.dirs = ['zp']

        self.gaussian_smoother = GaussianSmoothing(1, kernel_half_size, sigma/dx, 3)

        self.dx = dx
        

    def forward(self, d, v):      

        d = self.gaussian_smoother(d)
        
        vorticity = utils.torchJacobian3(v)[1] / self.dx

        v_norm = torch.sqrt(torch.sum(vorticity**2, dim=1)).unsqueeze(1)
        vel_norm = torch.sqrt(torch.sum(v**2, dim=1)).unsqueeze(1)
        #v_norm = vel_norm
        v_norm = self.gaussian_smoother(v_norm)

        out = []
        for dr in self.dirs:
            #out = out + [self.vortex_render(rotate(v_norm/torch.max(v_norm), dr), rotate(v_norm, dr))]
            out = out + [self.vortex_render(rotate(d, dr), rotate(v_norm, dr))]
            
        # returns a list of tuples (sketch, toon, shade, combined)

        return out

    def accumulate(self, d, v_norm, c=5, patch_sketch_info = None):

        # camera looks toward -z so we need to invert the z axsis
        c = 20

        d = torch.flip(d, [2])
        v_norm = torch.flip(v_norm, [2])

        x = torch.cumsum(d*self.dx, dim = 2)

        t = (x * c + 1) * torch.exp(-c*x)
        iv_norm = integrate(t, v_norm)
        
        # compute mask
        a = torch.clamp(integrate(t, torch.ones_like(d, requires_grad=False)), 0, 1)
 
        return a, iv_norm


    def vortex_render(self, d, v_norm):
        '''
        render sketch from a zp
        '''
        extinct_scale = 5
        
        # reuse gradient and normals
        a, iv_norm = self.accumulate(d, v_norm, c=extinct_scale)

        return  torch.clamp(iv_norm, 0, 1)

        
##################### Patch information extraction from coarse volume #################################

class CoarseVolumeHandler:

    def __init__(self, device, coarse_volume, dx):

        """
        Warning: dx is the dx of the *fine scale* patch
        """

        self.dx = dx
        self.device = device
        gaussian_smoother = GaussianSmoothing(1, 3, 1.6/self.dx, 3).to(device)
        self.coarse_volume = gaussian_smoother(patch.torch_upsample((coarse_volume+1)/2))
        #self.coarse_volume = gaussian_smoother(coarse_volume)
        self.coarse_grad   = -torchGrad3D(self.coarse_volume)/self.dx

    def get_patch_info_dir(self, dr, patch_pos, patch_size, c):

        """
        patch_size: patch size
        dr: direction ("xp", "xn", ...)
        dx: size of one cell in the patch
        c: sketching constant (need to be consitant with the sketcher)

        """
        upscaled_volume_size = np.array(self.coarse_volume.size()[2:])
        #volume_size = (upscaled_volume_size//2-1)+1
        volume_size = upscaled_volume_size

        pos = rotate_patch_pos(patch_pos, dr, volume_size)

        upscaled_pos = pos
        #upscaled_pos = pos//2
        upscaled_offset = (np.array([patch_size]*3)-1)//2
        #upscaled_offset = (np.array([patch_size]*3)-1)//4

        subvolume =  (rotate(self.coarse_volume, dr)[:, :, :, #keep all depth
                    upscaled_pos[1] - upscaled_offset[1]: upscaled_pos[1] + upscaled_offset[1] + 1, 
                    upscaled_pos[2] - upscaled_offset[2]: upscaled_pos[2] + upscaled_offset[2] + 1 ])

        subgrad = (rotate_grad(self.coarse_grad, dr)[:, :, :, #keep all depth
                    upscaled_pos[1] - upscaled_offset[1]: upscaled_pos[1] + upscaled_offset[1] + 1, 
                    upscaled_pos[2] - upscaled_offset[2]: upscaled_pos[2] + upscaled_offset[2] + 1 ])
        
        # integrate from 0 to t0
        
        if  upscaled_pos[0] + upscaled_offset[0] < subvolume.size(2)-1: #pre_d.size(2) > 1:

            pre_d = torch.flip(subvolume[:, :, upscaled_pos[0] + upscaled_offset[0]:], [2])
            pre_grad = torch.flip(subgrad[:, :, upscaled_pos[0] + upscaled_offset[0]:], [2])

            #pre_d[:, :, :-1] *= 2*self.dx
            # last slice of pre_d is overlaping betw coarse and fine. When doing the difference, one half of dx is missing
            #pre_d[:, :, -1] *= .5*self.dx

            pre_d[:, :, :-1] *= self.dx
            pre_d[:, :, -1] *= .25*self.dx

            x = torch.cumsum(pre_d, dim=2)
            t = (1 + c*x)*torch.exp(-c*x)

            x0 = x[:, :, -1]
            grad0 = integrate(t, pre_grad)
            a0 = integrate(t, torch.ones_like(t, requires_grad=False))
            #x0 = patch.torch_upsample(x[:, :, -1])
            #grad0 = patch.torch_upsample(integrate(t, pre_grad))
            #a0 = patch.torch_upsample(integrate(t, torch.ones_like(t, requires_grad=False)))
            
        else:
            # x0 = 0
            # grad0 = 0
            # a0 = 0
            x0 = torch.zeros((self.coarse_volume.size(0), 1, 33, 33), requires_grad=False, device=self.device)
            grad0 = torch.zeros((self.coarse_volume.size(0), 3, 33, 33), requires_grad=False, device=self.device)
            a0 = torch.zeros((self.coarse_volume.size(0), 1, 33, 33), requires_grad=False, device=self.device)

        # intergate from t1 to inf
        if upscaled_pos[0] - upscaled_offset[0]+1 > 1: #post_d.size(2) > 1:

            post_d = torch.flip(subvolume[:, :, :upscaled_pos[0] - upscaled_offset[0]+1], [2])
            post_grad = torch.flip(subgrad[:, :, :upscaled_pos[0] - upscaled_offset[0]+1], [2])

            #post_d[0] *= .5*self.dx
            #post_d[1:]*=2*self.dx
            post_d[0] *= .25*self.dx
            post_d[1:]*=1*self.dx

            x = torch.cumsum(post_d, dim=2)
            tt1 = torch.exp(-c*x)
            tt2 = (1 + c*x)*tt1
            #grad1 = patch.torch_upsample(integrate(tt1, post_grad))
            #grad2 = patch.torch_upsample(integrate(tt2, post_grad))

            #a1 = patch.torch_upsample(integrate(tt1, torch.ones_like(tt1, requires_grad=False)))
            #a2 = patch.torch_upsample(integrate(tt2, torch.ones_like(tt2, requires_grad=False)))

            grad1 = integrate(tt1, post_grad)
            grad2 = integrate(tt2, post_grad)

            a1 = integrate(tt1, torch.ones_like(tt1, requires_grad=False))
            a2 = integrate(tt2, torch.ones_like(tt2, requires_grad=False))

        else:
            # grad1 = 0
            # grad2 = 0
            # a1 = 0
            # a2 = 0
            grad1 = torch.zeros((self.coarse_volume.size(0), 3, 33, 33), requires_grad=False, device=self.device)
            grad2 = torch.zeros((self.coarse_volume.size(0), 3, 33, 33), requires_grad=False, device=self.device)
            a1 = torch.zeros((self.coarse_volume.size(0), 1, 33, 33), requires_grad=False, device=self.device)
            a2 = torch.zeros((self.coarse_volume.size(0), 1, 33, 33), requires_grad=False, device=self.device)

        return x0, grad0, grad1, grad2, a0, a1, a2


    def get_patch_info(self, dirs, patch_pos, patch_size, c=5):
        
        return {
            dr: self.get_patch_info_dir(dr, patch_pos, patch_size, c)
            for dr in dirs
        }


