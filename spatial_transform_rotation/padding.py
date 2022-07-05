import torch
import math
import numpy as np
from spatial_transform_rotation.tensor_utils import spatialDimensions, is3D, is2D
import torch.nn.functional as F


def padding_needed(target_shape, x_dims, spatial_dim):
    """
    :param target_shape: includes batch and channel dimension
    :param target_shape: spatial dimensions (without batch and channel)
    :param spatial_dim: without taking into account batch and channel dimension
    :return:
    """
    pad = (target_shape[2 + spatial_dim] - x_dims[spatial_dim]) / 2.0
    assert (pad >= 0.0)
    pad_0 = math.floor(pad)
    pad_1 = math.ceil(pad)
    return [pad_0, pad_1]


def get_pad_size(org_dims, ratio):
    pad_axis = ratio * org_dims - org_dims
    pad_side = pad_axis / 2.0
    pad_side = int(math.ceil(pad_side))
    return pad_side


class ReplicationPadMargin:
    def __init__(self, org_dims, ratio, n_spatial_dims=3):
        pad_side = get_pad_size(org_dims, ratio)
        self.res_padded = int(org_dims + pad_side * 2)

        if n_spatial_dims == 3:
            self._pad = [pad_side, pad_side, pad_side, pad_side, pad_side, pad_side]
        elif n_spatial_dims == 2:
            self._pad = [pad_side, pad_side, pad_side, pad_side]
        else:
            raise NotImplementedError()

    def __call__(self, x):
        return torch.nn.functional.pad(x, pad=self._pad, mode='replicate')


def pad_match_shape(target_shape, x, mode='constant'):
    x_dims = spatialDimensions(x)
    pad0 = padding_needed(target_shape, x_dims, 0)
    pad1 = padding_needed(target_shape, x_dims, 1)
    pad2 = padding_needed(target_shape, x_dims, 2) if len(x_dims) > 2 else []
    pad = list(reversed(pad0 + pad1 + pad2))
    padded = torch.nn.functional.pad(x, pad=pad, mode=mode)
    return padded


def add_fixed_pad(x, size, mode):
    if is3D(x):
        padded = F.pad(x, [
            size, size,
            size, size,
            size, size], mode=mode)
    elif is2D(x):
        padded = F.pad(x, [
            size, size,
            size, size], mode=mode)
    else:
        raise NotImplementedError()
    return padded


def pad_match(x1, x2, mode):
    if is3D(x1):
        # input is CDHW
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[4] - x1.size()[4]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [
            diffZ // 2, diffZ - diffZ // 2,
            diffX // 2, diffX - diffX // 2,
            diffY // 2, diffY - diffY // 2], mode=mode)
    elif is2D(x1):
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [
            diffY // 2, diffY - diffY // 2,
            diffX // 2, diffX - diffX // 2], mode=mode)
    else:
        raise NotImplementedError()
    return x1


def pad_zeros(target_shape, x):
    return pad_match_shape(target_shape, x, mode='constant')


def pad_zeros_end_dim(wanted_size, x, dim=2):
    pad = wanted_size - x.size(dim)
    need_shape = list(x.shape)
    need_shape[dim] = pad
    z = torch.zeros(need_shape, device=x.device)
    padded = torch.cat([x, z], dim=dim)
    return padded


def pad_zeros_into_square(x, exclude_dim={}):
    x_dims = spatialDimensions(x)
    max_dim = np.max(np.array(x_dims))
    target_shape = list(x.shape)
    for i in range(2, len(target_shape)):
        if i not in exclude_dim:
            target_shape[i] = max_dim
    return pad_zeros(target_shape, x)


def remove_pad(x, pad_size):
    if is2D(x):
        c = x[:,:, pad_size:-pad_size, pad_size:-pad_size]
    elif is3D(x):
        c = x[:, :, pad_size:-pad_size, pad_size:-pad_size, pad_size:-pad_size]
    else:
        raise NotImplementedError()
    return c


def remove_pad_list(x, pad_size):
    if is2D(x):
        if pad_size[1] == 0:
            pad_size[1] = -x.size(2)
        if pad_size[3] == 0:
            pad_size[3] = -x.size(3)
        c = x[:,:, pad_size[0]:-pad_size[1], pad_size[2]:-pad_size[3]]
    elif is3D(x):
        c = x[:, :, pad_size[0]:-pad_size[1], pad_size[2]:-pad_size[3], pad_size[4]:-pad_size[5]]
    else:
        raise NotImplementedError()
    return c


class FixedPad:
    def __init__(self, pad_size, mode):
        self._s = pad_size
        self._m = mode

    def size(self):
        return self._s

    def __call__(self, x):
        return self.add(x)

    def add(self, x):
        return add_fixed_pad(x, size=self._s, mode=self._m)

    def remove(self, x):
        return remove_pad(x, pad_size=self._s)


class PadPerAxis:
    def __init__(self, pad_size, mode):
        self._s = []
        for p in pad_size:
            self._s.append(math.floor(p*0.5))
            self._s.append(math.ceil(p*0.5))
        self._mode = mode

    def __call__(self, x):
        return self.add(x)

    def add(self, x):
        padded = F.pad(x, self._s[::-1], mode=self._mode)
        return padded

    def remove(self, x):
        return remove_pad_list(x, pad_size=self._s)
