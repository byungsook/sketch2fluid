import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import ndimage, misc

# physically-based data augmentation: borrow from tempoGAN
def rotate90(d, v, rot):

    datum = np.concatenate((d,v),0)
    
    # print (datum.shape)
    for axes in rot:
        for c in range(datum.shape[0]):
            # print (c)
            datum[c,...] = np.rot90(datum[c,...], axes=axes)

        channels = np.split(datum, datum.shape[0], 0)
        for v in [[1,2,3]]: #axes z,y,x -> vel x,y,z: 0,1,2 -> 2,1,0
            channels[v[-axes[0]+2]], channels[v[-axes[1]+2]] = -channels[v[-axes[1]+2]], channels[v[-axes[0]+2]]    
        datum = np.concatenate(channels, 0)
        # print (datum.shape)
    d = datum[0]
    v = datum[1:4]
    return d, v


def rotate90_d(d, rot):

    datum = d
    
    # print (datum.shape)
    for axes in rot:
        for c in range(datum.shape[0]):
            # print (c)
            datum[c,...] = np.rot90(datum[c,...], axes=axes)

        # channels = np.split(datum, datum.shape[0], 0)
        # for v in [[1,2,3]]: #axes z,y,x -> vel x,y,z: 0,1,2 -> 2,1,0
        #     channels[v[-axes[0]+2]], channels[v[-axes[1]+2]] = -channels[v[-axes[1]+2]], channels[v[-axes[0]+2]]    
        # datum = np.concatenate(channels, 0)
        # print (datum.shape)
    d = datum[0]
    return d

def rotate90_v(d, v, rot):

    datum = np.concatenate((d,v),0)
    c_list = [1,2,3]
    # print (datum.shape)
    for axes in rot:
        for c in c_list:
            # print (c)
            datum[c,...] = np.rot90(datum[c,...], axes=axes)

        channels = np.split(datum, datum.shape[0], 0)
        for v in [c_list]: #axes z,y,x -> vel x,y,z: 0,1,2 -> 2,1,0
            channels[v[-axes[0]+2]], channels[v[-axes[1]+2]] = -channels[v[-axes[1]+2]], channels[v[-axes[0]+2]]    
        datum = np.concatenate(channels, 0)
        # print (datum.shape)
    v = datum[1:4]
    return v



def scale_d(d, factor):

    datum = d
    scale = [1, factor, factor, factor] #single frame
    res_cube = 129
    # Zooming out

    # print (np.linspace(0.85, 1.15, 50))
    # for factor in np.linspace(0.85, 1.15, 1):
        # print (factor)
        # scale = [1, factor, factor, factor] #single frame
    if factor < 1:

        # Bounding box of the zoomed-out image within the output array
        z = int(np.round(res_cube * factor))
        c = (res_cube - z) // 2
        # print (z,c)
        out = np.zeros_like(datum)
        # print (out.shape, datum.shape)
        out[:, c:c+z, c:c+z, c:c+z] = ndimage.zoom(datum, scale, order=0, mode='constant', cval=0.0)

    # Zooming in
    elif factor > 1:
        z = int(np.round(res_cube / factor))
        c = (res_cube - z) // 2
        
        out = ndimage.zoom(datum[:, c:c+z,c:c+z, c:c+z], scale, order=0, mode='constant', cval=0.0)
        # print (out.shape)
        # trim_top = ((out.shape[1] - res_cube) // 2)
        # trim_left = ((out.shape[2] - w) // 2)
        # trim_depth = ((out.shape[2] - w) // 2)
        pad = res_cube-out.shape[-1]
        if pad > 0:
            out = np.pad(out, ((0,0), (0,pad), (0,pad), (0,pad)), 'constant')
        if pad < 0:
            out = out[:,0:res_cube,0:res_cube,0:res_cube]
        
        # out = out[trim_top:trim_top+h, trim_left:trim_left+w]
    # print ('d:', factor, out.shape)
    datum = out
    
    d_zoom = datum[0]
    # print (d_zoom.shape, d.shape)
    # plt.figure(1)
    # plt.subplot(121)
    # plt.imshow(d_zoom[64], cmap='gray')
    # plt.subplot(122)
    # plt.imshow(d[0,64], cmap='gray')
    # plt.show()
    return d_zoom

def scale_v(d, v, factor):

    datum = np.concatenate((d,v),0)
    scale = [1, factor, factor, factor] #single frame
    res_cube = 129

    # for factor in np.linspace(0.85, 1.15, 50):
        # print (factor)
        # scale = [1, factor, factor, factor] #single frame
    if factor < 1:
        
        # Bounding box of the zoomed-out image within the output array
        z = int(np.round(res_cube * factor))
        c = (res_cube - z) // 2
        # out = np.zeros_like(datum[1:4,...])

        out1 = np.zeros_like(datum[1:2])
        out2 = np.zeros_like(datum[2:3])
        out3 = np.zeros_like(datum[3:4])
        
        out1[:, c:c+z, c:c+z, c:c+z] = ndimage.zoom(datum[1:2,...], scale, order=0, mode='constant', cval=0.0)
        out2[:, c:c+z, c:c+z, c:c+z] = ndimage.zoom(datum[2:3,...], scale, order=0, mode='constant', cval=0.0)
        out3[:, c:c+z, c:c+z, c:c+z] = ndimage.zoom(datum[3:4,...], scale, order=0, mode='constant', cval=0.0)
        
        out = np.concatenate((out1,out2,out3),0)
        # print (out1.shape)

    # Zooming in
    elif factor > 1:
        z = int(np.round(res_cube / factor))
        c = (res_cube - z) // 2
        
        out1 = ndimage.zoom(datum[1:2, c:c+z, c:c+z, c:c+z], scale, order=0, mode='constant', cval=0.0)
        out2 = ndimage.zoom(datum[2:3, c:c+z, c:c+z, c:c+z], scale, order=0, mode='constant', cval=0.0)
        out3 = ndimage.zoom(datum[3:4, c:c+z, c:c+z, c:c+z], scale, order=0, mode='constant', cval=0.0)

        out = np.concatenate((out1,out2,out3),0)

        pad = res_cube-out.shape[-1]
        if pad > 0:
            out = np.pad(out, ((0,0), (0,pad), (0,pad), (0,pad)), 'constant')
        if pad < 0:
            out = out[:,0:res_cube,0:res_cube,0:res_cube]

        # datum[1:4] = np.concatenate((out1,out2,out3),0)
        # print (datum.shape)

    datum[1:4] = out

    # print ('v:', factor, datum.shape)
    c_list = [1,2,3]
    channels = np.split(datum, datum.shape[0], 0)
    for cv in [c_list]: # x,y,[z]; 2,1,0
        channels[cv[0]] *= factor
        channels[cv[1]] *= factor
        channels[cv[2]] *= factor
      
    datum = np.concatenate(channels, 0)
    # print (datum.shape)
    v_zoom = datum[1:4]

    # print (v_zoom.shape)
    # print (v.shape, v_zoom.shape)
    # plt.figure(1)
    # plt.subplot(121)
    # plt.imshow(v_zoom[0,64], cmap='gray')
    # plt.subplot(122)
    # plt.imshow(v[0,64], cmap='gray')
    # plt.show()

    return v_zoom



def flip_d(d, axes, isFrame=True): #axes=list, flip multiple at once
    
    # if not isFrame:
    #     axes = np.asarray(axes) + np.ones(axes.shape)

    # flip tiles/frames
    datum = d
    # print (axes)
    for axis in axes:
        for c in range(d.shape[0]):
            d[c,...] = np.flip(d[c,...], axis)

    d = datum[0]
    # print (d.shape)
    return d
    # data = self.special_aug(data, AOPS_KEY_FLIP, axes)


def flip_v(d, v, axes, isFrame=True): #axes=list, flip multiple at once
    
    # if not isFrame:
    #     axes = np.asarray(axes) + np.ones(axes.shape)

    # flip tiles/frames
    c_list = [1,2,3]
    datum = np.concatenate((d,v),0)
    # print (axes, datum.shape)

    for axis in axes:
        for v in [c_list]:
            datum[v[0],...] = np.flip(datum[v[0],...], axis)
            datum[v[1],...] = np.flip(datum[v[1],...], axis)
            datum[v[2],...] = np.flip(datum[v[2],...], axis)

    channels = np.split(datum, datum.shape[0], 0)
    for v in [c_list]:  # x,y,[z], 2,1,0
        if 2 in axes:   # flip vel x
            channels[v[0]] *= (-1)
        if 1 in axes:
            channels[v[1]] *= (-1)
        if 0 in axes and len(v)==3:
            channels[v[2]] *= (-1)

    datum = np.concatenate(channels, 0)
    v = datum[1:4]
    return v