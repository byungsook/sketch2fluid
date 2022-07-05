import torch
import numpy as np
import math

import spatial_transform_rotation.spatial_transformer as spatial_transformer

def yp_from_light(l):
    light_dir = np.array(l)
    light_dir = light_dir / np.sqrt(np.sum(light_dir**2))

    y = -math.asin(light_dir[0])
    if light_dir[2] < 0:
        y = -math.pi-y

    y = y*180/math.pi

    p = math.asin(light_dir[1])*180/math.pi

    return y,p


class Renderer(torch.nn.Module):

    def __init__(self, device, shape=[1, 1, 129, 129, 129]):

        super().__init__()

        assert shape[2] == shape[3] and shape[3] == shape[4]

        self.shape = shape

        self.scale_factor = 3
        self.large_size = math.ceil(shape[2]*math.sqrt(self.scale_factor))
        self.large_shape = [shape[0], shape[1], self.large_size, self.large_size, self.large_size]
        self.device = device
        self.transformer = spatial_transformer.SpatialTransformer(device, self.large_shape)


    def forward(self, d, lights=None, c=.2, background = 0):

        #d shape [B, 1, D, H, W]
        #tensor axis follow real world axis, meaning view is positive z
        #lights: iterable of dicts {'dir', 'rbg'}, pos will be normalized

        ### first pass: lights

        # assert np.all(np.array(d.shape) == np.array(self.shape))
        bs = d.size(0)

        light_array = d.new_zeros(size=[v if i !=1 else 3 for i, v in enumerate(d.shape)])

        if lights is None:
            lights = [{'dir': (2,1,1), 'rgb': (1,69/255,25/255)},
                    {'dir': (-1,.5,0), 'rgb': (227/255,255/255,66/255)}
            ]

        if background == 'default':
            background = d.new_tensor([96/255, 109/255, 138/255]).reshape((1,3,1,1))

        for l in lights:
            
            y, p = yp_from_light(l['dir'])

            # upsample to avoid rotation issues:
            up = d.new_zeros(self.large_shape).repeat(bs,1,1,1,1)
            i = self.large_size//2-d.shape[2]//2
            j = i+d.shape[2]
            up[:, :, i:j, i:j, i:j] = d

            # rotate to face light direction
            up = self.transformer(up, yaw=y, pitch=p, order="ypr")
            
            # accumulate light:
            t = torch.flip(torch.cumsum(torch.flip(up, [2]), dim = 2), [2])
            up = torch.exp(-c*t)

            # rotate back
            up = self.transformer(up, yaw=-y, pitch=-p, order="rpy")

            # get light color
            color = d.new_tensor(l['rgb']).reshape((1,3,1,1,1))

            # extract original volume
            light_array += up[:, :, i:j, i:j, i:j]*color

        # compute outcoming transmitance
        t = torch.exp(-c*torch.flip(torch.cumsum(torch.flip(d, [2]), dim = 2), [2]))

        # return render
        out = c*torch.sum(d*light_array*t, dim=2) + background*t[:,:, 0]
        
        return torch.clamp(out, 0, 1)




class HDRenderer(torch.nn.Module):

    def __init__(self, device, shape=[1, 1, 129, 129, 129], resolution=1024, perspective = 1.5):

        super().__init__()

        assert shape[2] == shape[3] and shape[3] == shape[4]

        self.shape = shape
        self.resolution = resolution
        self.perspective = perspective

        self.scale_factor = 3
        self.large_size = math.ceil(shape[2]*math.sqrt(self.scale_factor))
        self.large_shape = [shape[0], shape[1], self.large_size, self.large_size, self.large_size]
        self.device = device
        self.transformer = spatial_transformer.SpatialTransformer(device, self.large_shape)


    def forward(self, d, lights=None, c=.2, background = 0):

        #d shape [B, 1, D, H, W]
        #tensor axis follow real world axis, meaning view is positive z
        #lights: iterable of dicts {'dir', 'rbg'}, pos will be normalized

        ### first pass: lights

        # assert np.all(np.array(d.shape) == np.array(self.shape))
        bs = d.size(0)

        light_array = d.new_zeros(size=[v if i !=1 else 3 for i, v in enumerate(d.shape)])

        if lights is None:
            lights = [{'dir': (2,1,1), 'rgb': (1,69/255,25/255)},
                    {'dir': (-1,.5,0), 'rgb': (227/255,255/255,66/255)}
            ]

        d = torch.clamp(d, 0, 1)

        if type(background) == str and background == 'default':
            background = d.new_tensor([96/255, 109/255, 138/255]).reshape((1,3,1,1))

        elif torch.all(torch.tensor(background) == 1):
            background = d.new_tensor([255/255, 255/255, 255/255]).reshape((1,3,1,1))

        for l in lights:
            
            y, p = yp_from_light(l['dir'])

            # upsample to avoid rotation issues:
            up = d.new_zeros(self.large_shape).repeat(bs,1,1,1,1)
            i = self.large_size//2-d.shape[2]//2
            j = i+d.shape[2]
            up[:, :, i:j, i:j, i:j] = d

            # rotate to face light direction
            up = self.transformer(up, yaw=y, pitch=p, order="ypr")
            
            # accumulate light:
            t = torch.flip(torch.cumsum(torch.flip(up, [2]), dim = 2), [2])
            up = torch.exp(-c*t)

            # rotate back
            up = self.transformer(up, yaw=-y, pitch=-p, order="rpy")

            # get light color
            color = d.new_tensor(l['rgb']).reshape((1,3,1,1,1))

            # extract original volume
            light_array += up[:, :, i:j, i:j, i:j]*color

        ## resample

        d_hd = d.new_zeros((self.shape[0], 1, self.shape[2], self.resolution, self.resolution))
        l_hd = d.new_zeros((self.shape[0], 3, self.shape[2], self.resolution, self.resolution))

        light_array = light_array * d

        for i in range(self.shape[2]):
            x = i/(self.shape[2]-1) #x = 0 is far
            target_res = int(self.resolution * ((1-x)/self.perspective+x))
            s = (self.resolution -target_res)//2
            r = s + target_res
            d_hd[:, :, i, s:r, s:r] = torch.nn.functional.interpolate(
                d[:, :, i], 
                size=target_res, 
                mode = 'bilinear', align_corners=False)
            l_hd[:, :, i, s:r, s:r] = torch.nn.functional.interpolate(
                light_array[:, :, i], 
                size=target_res, 
                mode = 'bilinear', align_corners=False)

        # compute outcoming transmitance
        t = torch.exp(-c*torch.flip(torch.cumsum(torch.flip(d_hd, [2]), dim = 2), [2]))

        # return render
        out = c*torch.sum(l_hd * t, dim=2) + background*t[:,:, 0]
        
        return torch.clamp(out, 0, 1)









class RendererFaster(torch.nn.Module):

    def __init__(self, device, scale_factor=1, batch_size=1, shape=[1, 1, 129, 129, 129]):

        super().__init__()

        assert shape[2] == shape[3] and shape[3] == shape[4]

        self.shape = shape

        self.scale_factor = scale_factor
        self.large_size = math.ceil(shape[2]*math.sqrt(self.scale_factor))
        self.large_shape = [shape[0], shape[1], self.large_size, self.large_size, self.large_size]
        
        self.batch_size = batch_size
        self.device = device
        self.transformer = spatial_transformer.SpatialTransformer(device, self.large_shape)
        self.light_array = torch.zeros((1,3,shape[-1],shape[-1],shape[-1]), requires_grad=False).to(self.device).repeat(batch_size,1,1,1,1)

    def forward(self, d, lights=2, c=.2, background = 'default'):

        #d shape [B, 1, D, H, W]
        #tensor axis follow real world axis, meaning view is positive z
        #lights: iterable of dicts {'dir', 'rbg'}, pos will be normalized

        ### first pass: lights
        
        if lights is None:
            lights = [
                {'dir': (2,1,1), 'rgb': (1,69/255,25/255)}
            ]

        if lights == 2:
            lights = [
                {'dir': (2,1,1), 'rgb': (1,69/255,25/255)},
                {'dir': (-1,.5,0), 'rgb': (227/255,255/255,66/255)}
            ]

        if background == 'default':
            background = d.new_tensor([96/255, 109/255, 138/255]).reshape((1,3,1,1))

        for l in lights:
            
            y, p = yp_from_light(l['dir'])

            i = self.large_size//2-d.shape[2]//2
            j = i+d.shape[2]

            if self.scale_factor > 1:
                up = d.new_zeros(self.large_shape).repeat(self.batch_size,1,1,1,1)
                up[:, :, i:j, i:j, i:j] = d
            else:
                up = d

            up = self.transformer(up, yaw=y, pitch=p, order="ypr")

            # accumulate light:
            t = torch.flip(torch.cumsum(torch.flip(up, [2]), dim = 2), [2])
            up = torch.exp(-c*t)
            # rotate back
            up = self.transformer(up, yaw=-y, pitch=-p, order="rpy")

            # get light color
            color = d.new_tensor(l['rgb']).reshape((1,3,1,1,1))
            # extract original volume
            
            if self.scale_factor > 1:
                self.light_array += up[:, :, i:j, i:j, i:j]*color
            else:
                self.light_array += up*color

            
        # compute outcoming transmitance
        t = torch.exp(-c*torch.flip(torch.cumsum(torch.flip(d, [2]), dim = 2), [2]))

        # return render
        out = c*torch.sum(d*self.light_array*t, dim=2) + background*t[:,:, 0]
        
        self.light_array[:] = 0 # reset

        return torch.clamp(out, 0, 1)

