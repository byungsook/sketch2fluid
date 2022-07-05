from collections import namedtuple
import torch
import os
import math
#from utils.py_utils import null
from random import randint

Extent2D = namedtuple('Dims', 'H W')
Extent3D = namedtuple('Dims', 'D H W')


def false(x):
    return (type(x) == bool and not x) or \
           (isinstance(x, torch.Tensor) and x.ndim == 1 and x.size(0) == 1 and not x.item())


def get_item(x):
    if isinstance(x, torch.Tensor):
        return x.item()
    else:
        return x


# [B,C,H,W]
def is2D(tensor):
    return tensor.dim() == 4


# [B,C,D,W,H]    
def is3D(tensor):
    return tensor.dim() == 5


def spatialDimensions(tensor):
    if is2D(tensor):
        return Extent2D(tensor.size(2), tensor.size(3))

    elif is3D(tensor):
        return Extent3D(tensor.size(2), tensor.size(3), tensor.size(4))


def n_spatial_dimensions(tensor):
    return tensor.ndim - 2


def spatialSize(tensor):
    dims = spatialDimensions(tensor)
    total_size = 1.0
    for ds in dims:
        total_size *= ds
    return total_size


def isTensor(x):
    return "Tensor" in str(type(x))


def add_thickness(x, thickness=None):
    """
    Adds depth dimension to convert 2D tensor into 3D tensor
    """
    assert (is2D(x))
    _W, _H = spatialDimensions(x)
    x = x.unsqueeze(2)
    _D = thickness if not null(thickness) else min(_W, _H)
    x = torch.repeat_interleave(x, _D, 2)
    return x


def expand_channels(x, n_chan, pos='after'):
    assert (is3D(x))
    remaining = n_chan - x.size(1)
    zeros = torch.zeros_like(x[:, 0:1])
    zeros = torch.repeat_interleave(zeros, 1, remaining)
    if pos == 'after':
        x = torch.cat([x, zeros], dim=1)
    elif pos == 'before':
        x = torch.cat([zeros, x], dim=1)
    else:
        raise NotImplementedError()
    return x


class To3DMod(torch.nn.Module):
    def __init__(self, expand_channels, decay, depth=None):
        super(To3DMod, self).__init__()
        self._ec = expand_channels
        self._decay = decay
        self._depth = depth

    @staticmethod
    def _expand_channels(x):
        assert (is3D(x))
        remaining = 3 - x.size(1)
        zeros = torch.zeros_like(x[:, 0:1])
        zeros = torch.repeat_interleave(zeros, 1, remaining)
        x = torch.cat([x, zeros], dim=1)
        return x

    def forward(self, x):
        if not isinstance(x, list):
            _x = [x]
        else:
            _x = x
        for i in range(len(_x)):
            _x[i] = add_thickness(_x[i], self._depth)

            if i < len(self._decay) and self._decay[i]:
                center = _x[i].size(2) // 2
                decay_strength = (1.0 / float(center)) * 5.0
                for j in range(_x[i].size(2)):
                    d = abs(center - j)
                    decay_j = math.exp(-d * decay_strength)
                    _x[i][:, :, j] *= decay_j

            if i < len(self._ec) and self._ec[i]:
                _x[i] = expand_channels(_x[i], n_chan=3, pos='after')

        if len(_x) == 1:
            _x = _x[0]

        return _x


def random_slice(tensor, dims=[2, 4], squeeze=False, swap_to_depth_dim=True):
    dim_idx = randint(0, len(dims) - 1)
    dim = dims[dim_idx]
    idx = randint(0, tensor.size(dim) - 1)
    idx = torch.tensor(idx, device=tensor.device)
    slice = torch.index_select(input=tensor, dim=dim, index=idx)
    sq_dim = dim
    if swap_to_depth_dim:
        slice = slice.transpose(2, dim)
        sq_dim = 2
    if squeeze:
        slice = slice.squeeze(sq_dim)
    return slice


def random_slice_with_thickness(tensor, dims=[2, 4], squeeze=False, swap_to_depth_dim=True, renderers=None):
    dim_idx = randint(0, len(dims) - 1)
    dim = dims[dim_idx]
    idx1 = randint(0, tensor.size(dim) - 2)
    idx2 = randint(idx1 + 1, tensor.size(dim) - 1)
    steps = idx2 - idx1
    idx = torch.linspace(idx1, idx2, steps=steps, device=tensor.device).long()
    slice = torch.index_select(input=tensor, dim=dim, index=idx)

    if swap_to_depth_dim:
        slice = slice.transpose(2, dim)

    if not null(renderers):
        r_idx = randint(0, len(renderers) - 1)
        slice = renderers[r_idx](slice)
        slice = slice.unsqueeze(2)  # make depth
    else:
        slice = torch.sum(slice, dim=2, keepdim=True)
        slice = slice / steps  # torch.max(slice)

    if squeeze:
        slice = slice.squeeze(dim)
    return slice
