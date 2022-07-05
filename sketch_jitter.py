import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np 

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


    def __init__(self, channels=1, kernel_half_size=3, sigma=1.6, dim=2):

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

        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=kernel_half_size)
        self.conv.weight.data.copy_(kernel)
        self.conv.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        
        self.conv_block.add_module("conv2d", self.conv)

    def forward(self, x):

        for layer in self.conv_block:
            x = layer(x)
        return x





class SketchJitter(nn.Module):
    """

    """
    def __init__(self, device, kernel_half_size=3, sigma=0.8):

        super(SketchJitter, self).__init__()

        '''
            Apply jittering on sketch:
                slur
                shift
                gaussian filter
                ...
            Input:
                sketch: (B,C,H,W), [0,1]
            Output:
                jittered sketch with same size
        '''

        # 2d gaussian filter for sketch

        if sigma == 0:
            sigma = 1e-4
        # print (sigma)
        self.device = device
        self.gaussian_smoother = GaussianSmoothing(1, kernel_half_size, sigma, 2).to(self.device)
        

    def shift(self, x, theta_translate):
        '''
            Not tested on GPU, migh need to set device
        '''
        theta = torch.zeros(x.size(0),2,3).to(self.device)
        theta[:, 0, 0] = 1
        theta[:, 1, 1] = 1
        # theta = Variable(theta, requires_grad=True)
        
        # theta_translation = self.a+self.b*torch.rand(1, 2)
        # theta[:, :, 2] = theta_translation

        # theta[:, :, 2] = theta_translation = torch.FloatTensor([-0,theta_translate]) # [-theta_translate,0]
        theta[:, :, 2] = torch.FloatTensor(theta_translate).to(self.device) # [-theta_translate,0]
        
        grid = F.affine_grid(theta, x.size())
        x1 = F.grid_sample(x, grid)

        # theta[:, :, 2] = theta_translation = torch.FloatTensor([theta_translate,0])
        # grid = F.affine_grid(theta, x.size())
        # x2 = F.grid_sample(x, grid)
        
        return 1-x1#, 1-x2

    def slur(self, x, theta_translate):
        y = 1 - x
        y1 = self.shift(y, theta_translate)
        y = (x + y1) / 2
        return y

    def brightness_contrast(self, x, brightness=127, contrast=64):

        '''
            Not tested with different combinations:
                blist = [0, -127, 127,   0,  0, 64] # list of brightness values
                clist = [0,    0,   0, -64, 64, 64] # list of contrast values
        '''
        
        x = x * 255.0  
        if brightness != 0:
            if brightness > 0:
                shadow = brightness
                highlight = 255
            else:
                shadow = 0
                highlight = 255 + brightness
            alpha_b = (highlight - shadow) / 255
            gamma_b = shadow

            buf = x * alpha_b + gamma_b
        else:
            buf = x

        if contrast != 0:
            f = 131*(contrast + 127)/(127*(131-contrast))
            alpha_c = f
            gamma_c = 127*(1-f)

            buf = buf * alpha_c + gamma_c

        

        buf = buf / 255.0
        # print (buf.min(), buf.max())
        
        buf = buf / buf.max()
        buf = torch.clamp(buf,0,1)
        return buf

    def blur(self, x):
        y = 1 - x
        y = self.gaussian_smoother(y)
        y = 1 - y
        return y

    def add_noise(self, x, std):    # std = ...
        noise = x.data.new(x.size()).normal_(0.0, std).to(self.device)
        x = x + noise
        x = x / x.max()
        # print (x.min(), x.max())
        return x

    def forward(self, x):  
        pass
        # blur = self.blur(x) # seems optional
        # slur = self.slur(x)
        # bc = self.brightness_contrast(x)
        # noise = self.add_noise(x)
        # return blur, slur, bc, noise

