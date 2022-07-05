#%%
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


################################################### "clean" pytorch functions ###################################################

def torch_upsample(patch, dim=None):

    """ 
    nd version of n-linear upsampling, on the dims 2,...
    """
    # print ('recur')
    shape = np.array(patch.size())

    if dim == shape.size-1 or dim ==-1:
        shape[-1] = 2*(shape[-1]-1)+1
        r = patch.new_empty(tuple(shape))
        r[...,::2]=patch
        r[...,1::2] = .5*(patch[...,:-1] + patch[...,1:])
        return r
    else:
        # print ('upsample')
        if dim is None:
            dim = tuple(range(2, shape.size))

        if isinstance(dim, tuple):
            tmp = patch
            for d in dim:
                tmp = torch_upsample(tmp, dim=d)
            return tmp
        else:
            return torch.transpose(torch_upsample(torch.transpose(patch, dim, -1), -1), dim, -1)

def torch_downsample(patch, factor = 2, dim=None):

    """ 
    nd version of n-linear upsampling, on the dims 2,...
    factor should be int and was only tested with powers of two
    """
    
    shape = np.array(patch.size())

    if dim == shape.size-1 or dim ==-1:
        return patch[..., ::factor]
    else:
        if dim is None:
            dim = tuple(range(2, shape.size))

        if isinstance(dim, tuple):
            tmp = patch
            for d in dim:
                tmp = torch_downsample(tmp, factor = factor, dim=d)
            return tmp
        else:
            return torch.transpose(torch_downsample(torch.transpose(patch, dim, -1), factor = factor, dim=-1), dim, -1)

def _torch_slice(data, slices):
    """
        Generalized slices, where slices is a list of tuples (dim, start, end)
        The slices are filled with zeros if start, end are out of bound
        As a result, negative index to designate the end of the array is not supported

        Return a view, unless padding is necessary, where it returns a copy
    """

    shape = data.shape
    target_shape = list(shape)

    for dim, s, e in slices:
        target_shape[dim] = e-s
    
    out = data
    pad = []
    for dim, s, e in slices:

        safe_s = max(0, s)
        safe_e = min(shape[dim], e)
        safe_len = safe_e-safe_s

        if safe_len <=0:
            return data.new_zeros(target_shape)
        
        out = out.narrow(dim, int(safe_s), int(safe_len))
        pad = pad + [(dim, max(0, -s), e-s - max(0, e - shape[dim]))]
 
    for dim, ps, pe in pad:
        if ps != 0 or pe != target_shape[dim]:
            tmp_shape = list(out.shape)
            tmp_shape[dim] = target_shape[dim]
            tmp = out.new_zeros(tuple(tmp_shape))
            tmp.narrow(dim, int(ps), int(pe-ps))[:] = out
            out = tmp

    return out

def _upsample_middle_patch_scale(patch_scale):
    """
    input: single scale of a patch (bs, channel, res1, res2, res3)
    output: extract the middle of this patch and upsmaple it
    output resolution shape is the same as input
    """

    sz = np.array(list(patch_scale.size()[2:])) # get (res1, ... )

    half = sz//2
    q = half//2
    q3 = half+q+1

    return torch_upsample(_torch_slice(patch_scale, [(2+dim, q[dim], q3[dim]) for dim in range(sz.size)]))


def torch_eval_patch(patch):
    '''
        pytorch implementation for evaluating patch supporting batch_size
        input: (bs, scale, channel, res1, res2, res3)
        output: (bs, channel, res1, res2, res3)
        supports 1D, 2D and 3D, with and without multiple channels (velocities)
    '''

    if patch.size(1) == 1:
        return patch[:,0] # remove scale dimension 

    coarse = torch_eval_patch(patch[:,1:]) # dim of coarse is (bs, channel, res1, ...)

    return _upsample_middle_patch_scale(coarse) + patch[:, 0]


def torch_residuals(non_residual_patch):

    """
    convert a non residual patch to residual patch
    outputs a copy
    """

    out = non_residual_patch.clone()

    #size 1 is the number of scales
    for s in range(0, out.size(1)-1):
        out[:, s] = out[:, s] - _upsample_middle_patch_scale(out[:, s+1])
    
    return out




def torch_get_patch(volume, p, pos, smaller_base_volume = True):
    """
    Extract multiscale patches
    volume: original 3D data, of shape N, C, (2**n+1)**ndim
    patch size: 2**p+1, p>n/2
    pos: center of the patch. Should be a multiple of 2**(n-p)

    if smaller_base_volume is True, will stop the patch resolution at a patch scale where the whole volume can be enclosed within the patch

    return a sequence of residual patches
    """

    #### sanity checks ####

    # only tested with cubic volumes (TODO: non cubic)
    for k in range(3, len(volume.shape)):
        assert volume.shape[k] == volume.shape[2]

    volume_size = volume.shape[2] # TODO: should be min if non cubic
    pos = np.array(pos)

    # safety check on the patch size. Assume that the volume shape is 2**n+1
    # condition: 2p > n
    assert 2**(2*p) > volume_size-1, "Patch size exponent should be strictly larger than log2(min volume shape-1)/2"

    #check the position
    assert (volume_size-1) / 2**p == (volume_size-1) // 2**p, "min volume shape-1 should be a multiple of 2**p"
    assert np.all(pos / ((volume_size-1) / 2**p) == pos / ((volume_size-1) // 2**p)), "position should be a multiple of (volume_size-1) / 2**p)"


    #### implem ####

    patch_shape = list(volume.shape)
    vol_check_mult = 2 if smaller_base_volume else 1
    
    for i in range(2, len(patch_shape)):
        patch_shape[i] = 2**p+1

    offset = np.array((2**(p-1),)*(len(volume.shape)-2))

    patches = []
    ds = 1
    while True:
        
        vol_ds = torch_downsample(volume, factor = ds)
        pos_ds = pos//ds
        patch = _torch_slice(vol_ds, [(dim+2, (pos_ds-offset)[dim], (pos_ds+offset+1)[dim]) for dim in range(offset.size)])
        patches = patches + [patch]

        if vol_check_mult*(vol_ds.shape[2]-1) <= 2**p:
            break

        ds *= 2

    non_residual = torch.stack(patches, dim=1)

    return non_residual
    # return torch_residuals(non_residual)

























############################################################## previous ##############################################################


def upsample3D(patch):
    # print (patch.shape)
    r = np.empty(2*(np.array(patch.shape)-1)+1, dtype = patch.dtype)
    # print (r.shape, patch.shape)
    r[::2, ::2, ::2] = patch
    r[1::2, ::2, ::2] = .5*(patch[:-1] + patch[1:])
    r[::2, 1::2, ::2] = .5*(patch[:, :-1] + patch[:, 1:])
    r[::2, ::2, 1::2] = .5*(patch[:, :, :-1] + patch[:, :, 1:])

    r[::2, 1::2, 1::2] = .25*(patch[:, :-1, :-1] + patch[:, :-1, 1:]+patch[:, 1:, :-1] + patch[:, 1:, 1:])
    r[1::2, ::2, 1::2] = .25*(patch[:-1, :, :-1] + patch[:-1, :, 1:]+patch[1:, :, :-1] + patch[1:, :, 1:])
    r[1::2, 1::2, ::2] = .25*(patch[:-1, :-1] + patch[:-1, 1:]+patch[1:, :-1] + patch[1:, 1:])
    r[1::2, 1::2, 1::2] = .125*(patch[:-1, :-1, :-1] + patch[:-1, :-1, 1:]+patch[:-1, 1:, :-1] + patch[:-1, 1:, 1:]+patch[1:, :-1, :-1] + patch[1:, :-1, 1:]+patch[1:, 1:, :-1] + patch[1:, 1:, 1:])

    # print (patch.shape, r.shape)
    return r

def torch_upsample3D(patch):
    raise Exception ("Outdated, use new function in patch.py")
    # print (patch.size())
    bs = patch.size(0)
    # print (bs)
    res = list(patch.size()[1:])
    dim = 2*(np.array(res)-1)+1
    # print ('dim:', bs, dim)
    if torch.cuda.is_available():
        r = torch.zeros((bs,dim[0],dim[1],dim[2]), device='cuda')
    else:
        r = torch.zeros((bs,dim[0],dim[1],dim[2]))
    # print (patch.size(), r.size())
    # print (r[:,::2, ::2, ::2].size())
    r[:,::2, ::2, ::2] = patch[:,...]
    r[:,1::2, ::2, ::2] = .5*(patch[:,:-1] + patch[:,1:])
    r[:,::2, 1::2, ::2] = .5*(patch[:,:, :-1] + patch[:,:, 1:])
    r[:,::2, ::2, 1::2] = .5*(patch[:,:, :, :-1] + patch[:,:, :, 1:])

    r[:,::2, 1::2, 1::2] = .25*(patch[:,:, :-1, :-1] + patch[:,:, :-1, 1:]+patch[:,:, 1:, :-1] + patch[:,:, 1:, 1:])
    r[:,1::2, ::2, 1::2] = .25*(patch[:,:-1, :, :-1] + patch[:,:-1, :, 1:]+patch[:,1:, :, :-1] + patch[:,1:, :, 1:])
    r[:,1::2, 1::2, ::2] = .25*(patch[:,:-1, :-1] + patch[:,:-1, 1:]+patch[:,1:, :-1] + patch[:,1:, 1:])
    r[:,1::2, 1::2, 1::2] = .125*(patch[:,:-1, :-1, :-1] + patch[:,:-1, :-1, 1:]+patch[:,:-1, 1:, :-1] + patch[:,:-1, 1:, 1:]+patch[:,1:, :-1, :-1] + patch[:,1:, :-1, 1:]+patch[:,1:, 1:, :-1] + patch[:,1:, 1:, 1:])

    # print (r.size())
    return r

def upsample2D(patch):
    r = np.empty(2*(np.array(patch.shape)-1)+1, dtype = patch.dtype)
    r[::2, ::2] = patch
    r[1::2, ::2] = .5*(patch[:-1] + patch[1:])
    r[::2, 1::2] = .5*(patch[:, :-1] + patch[:, 1:])
    r[1::2, 1::2] = .25*(patch[:-1, :-1] + patch[:-1, 1:]+patch[1:, :-1] + patch[1:, 1:])

    return r


def torch_upsample2D(patch):
    raise Exception ("Outdated, use new function in patch.py")
    # r = np.empty(2*(np.array(patch.shape)-1)+1, dtype = patch.dtype)
    bs = patch.size(0)
    # print (bs)
    res = list(patch.size()[1:])
    dim = 2*(np.array(res)-1)+1
    # print ('dim:', bs, dim)
    if torch.cuda.is_available():
        r = torch.zeros((bs,dim[0],dim[1]), device='cuda')
    else:
        r = torch.zeros((bs,dim[0],dim[1]))

    r[:,::2, ::2] = patch[:,...]
    r[:,1::2, ::2] = .5*(patch[:,:-1] + patch[:,1:])
    r[:,::2, 1::2] = .5*(patch[:,:, :-1] + patch[:,:, 1:])
    r[:,1::2, 1::2] = .25*(patch[:,:-1, :-1] + patch[:,:-1, 1:]+patch[:,1:, :-1] + patch[:,1:, 1:])

    return r

def ob_slice(data, begin, end):

    """
    Extract the slice data[zip(being:end)], filling with 0 if out of bound
    """

    begin = np.array(begin)
    end = np.array(end)

    # patch shape
    p_shape = end-begin

    #data shape
    d_shape = np.array(data.shape)

    #slice extremum in data space
    d_begin = np.maximum(begin, 0)
    d_end = np.minimum(end, d_shape)

    #slice extremum in patch space
    p_begin = np.maximum(-begin, 0)
    p_end = p_shape - np.maximum(0, end - d_shape)

    #output
    patch = np.zeros(p_shape, dtype = data.dtype)

    if np.all(p_begin<p_shape) and np.all(p_end > 0):
        if len(data.shape) == 2:
            patch[p_begin[0]:p_end[0], p_begin[1]:p_end[1]] = data[d_begin[0]:d_end[0], d_begin[1]:d_end[1]]
        elif len(data.shape) == 3:
            patch[p_begin[0]:p_end[0], p_begin[1]:p_end[1], p_begin[2]:p_end[2]] = data[d_begin[0]:d_end[0], d_begin[1]:d_end[1], d_begin[2]:d_end[2]]
        else:
            assert False, "Arrays of dimension {} are not supported".format(len(data.size))

    return patch


def get_patch(volume, p, pos, smaller_base_volume = True):
    """
    Extract multiscale patches
    volume: original 3D data, of size 2**n+1
    patch size: 2**p+1, p>n/2
    pos: center of the patch. Should be a multiple of 2**(n-p)

    if smaller_base_volume is True, will stop the patch resolution at a patch scale where the whole volume can be enclosed within the patch

    return a sequence of residual patches
    """

    # only tested with cubic volumes (TODO: non cubic)
    assert volume.shape[0] == volume.shape[1] and (len(volume.shape)==2 or volume.shape[0] == volume.shape[2])
    volume_size = np.min(np.array(volume.shape))

    # safety check on the patch size. Assume that the volume shape is 2**n+1
    # condition: 2p > n
    assert 2**(2*p) > volume_size-1, "Patch size exponent should be strictly larger than log2(min volume shape-1)/2"

    #check the position
    assert (volume_size-1) / 2**p == (volume_size-1) // 2**p, "min volume shape-1 should be a multiple of 2**p"
    assert np.all(pos / ((volume_size-1) / 2**p) == pos / ((volume_size-1) // 2**p)), "position should be a multiple of (volume_size-1) / 2**p)"

    return _get_patch(volume, p, pos, 2 if smaller_base_volume else 1)[0]


def _get_patch(volume, p, pos, vol_check_mult):

    """
    return patch, patch_eval
    """
    
    # extract the patch at 1:1 scale
    offset = np.array((2**(p-1),)*len(volume.shape))
    # print (pos, offset)
    patch = ob_slice(volume, pos-offset, pos+offset+1)
    # print (patch.shape)

    # if this is the smallest volume size, return it
    if vol_check_mult*(np.min(np.array(volume.shape))-1) <= 2**p:
        return patch[None, ...], patch

    # else get the larger scale patches
    coarse, coarse_eval = _get_patch(
        volume[::2, ::2, ::2] if len(volume.shape) == 3 else volume[::2, ::2],
        p, pos//2, vol_check_mult)

    # compute the residual
    center = offset
    offset = np.array((2**(p-2),)*len(volume.shape))
    if len(volume.shape) == 2:
        coarse_upsample = upsample2D(ob_slice(coarse_eval, center-offset, center+offset+1))
    else:
        # print ( (ob_slice(coarse_eval, center-offset, center+offset+1)).shape )
        coarse_upsample = upsample3D(ob_slice(coarse_eval, center-offset, center+offset+1))
        # print (coarse_upsample.shape)

    patch_eval = patch.copy()
    patch -= coarse_upsample
    
    # print ( (np.concatenate([patch[None, ...], coarse])).shape )
    return np.concatenate([patch[None, ...], coarse]), patch_eval


def eval_patch(patch):
     
    if patch.shape[0] == 1:
        return patch[0]

    sz = np.array(patch[0].shape)
    # print (sz)
    half = sz//2
    q = half//2
    q3 = half+q+1
    if sz.size == 2:
        coarse = eval_patch(patch[1:])[q[0]:q3[0], q[1]:q3[1]]
        return upsample2D(coarse) + patch[0]
    else:
        coarse = eval_patch(patch[1:])[q[0]:q3[0], q[1]:q3[1], q[2]:q3[2]]
        return upsample3D(coarse) + patch[0]


def torch_eval_all_patch_d(patch_residual):
    raise Exception ("Outdated, use new function in patch.py")
    '''
        pytorch implementation for evaluating patch supporting batch_size
        input: (bs, scale, z, y, x)
        notice: only supports 3D for now
    '''
    patch_residual = patch_residual[:,:,0,...]  # remove channel
    # print (patch_residual.size())
    p3 = patch_residual[:,3]    # keep batch_size
    p3 = (p3 + 1) / 2
    # print (p3.shape, patch_residual[2].shape, np.stack((patch_residual[2], p3), 0).shape)
    p2 = torch_eval_patch( torch.stack( (patch_residual[:,2], p3), 1) )
    p1 = torch_eval_patch( torch.stack( (patch_residual[:,1], p2), 1) )
    p0 = torch_eval_patch( torch.stack( (patch_residual[:,0], p1), 1) )
    patch = torch.stack((p0, p1, p2, p3), 1)
    # print (patch.size(), patch_residual.size())
    # print (patch.size(), patch.min(), patch.max())
    patch = 2 * patch - 1
    patch = patch.unsqueeze(2)
    return patch


def _torch_eval_all_patch_v(patch_residual):
    raise Exception ("Outdated, use new function in patch.py")
    p3 = patch_residual[:,3]
    p2 = torch_eval_patch( torch.stack( (patch_residual[:,2], p3), 1) )
    p1 = torch_eval_patch( torch.stack( (patch_residual[:,1], p2), 1) )
    p0 = torch_eval_patch( torch.stack( (patch_residual[:,0], p1), 1) )
    patch = torch.stack((p0, p1, p2, p3), 1)
    # print (patch.size(), patch.min(), patch.max())
    return patch
def torch_eval_all_patch_v(patch_residual):
    raise Exception ("Outdated, use new function in patch.py")
    '''
        pytorch implementation for evaluating patch supporting batch_size
        input: (bs, scale, c, z, y, x)
        notice: only supports 3D for now
    '''
    # print (patch_residual.size())
    patch_u = _torch_eval_all_patch_v(patch_residual[:,:,0,...])  # for each channel
    patch_v = _torch_eval_all_patch_v(patch_residual[:,:,1,...])  # for each channel
    patch_w = _torch_eval_all_patch_v(patch_residual[:,:,2,...])  # for each channel
    
    patch = torch.stack((patch_u, patch_v, patch_w), 2)
    # print (patch.size(), patch.min(), patch.max())
    
    return patch

def _load_patch_s(x, p, pos):
    patch_residual = get_patch(x, p, pos)

    p3 = patch_residual[3]
    p2 = eval_patch( np.stack( (patch_residual[2], p3), 0) )
    p1 = eval_patch( np.stack( (patch_residual[1], p2), 0) )
    p0 = eval_patch( np.stack( (patch_residual[0], p1), 0) )

    patch_x = np.stack((p0, p1, p2, p3), 0)
    # patch_x = patch_residual

    patch_x = 2 * patch_x - 1
    # print (patch_x.shape, patch_x.min(), patch_x.max())
    # if self.activ_d == 'tanh':
    

    return patch_x

def load_patch_s(x1, x2, pos, patch_den=None, p=5):

    # print (x1.shape, x2.shape, pos.shape)
    x1 = (x1 + 1) / 2
    x2 = (x2 + 1) / 2
    # print (x1.shape, x2.shape, pos.shape)
    # print (x1.min(), x1.max(), x2.min(), x2.max())
    assert x1.shape[0] == x2.shape[0] and x1.shape[0] == pos.shape[0]
    
    # x1 = np.pad(x1, ((0,0), (0,0), (0, 1), (0, 1)), 'constant')
    # x2 = np.pad(x2, ((0,0), (0,0), (0, 1), (0, 1)), 'constant')
    
    # print (x1.shape, x2.shape)
    # import matplotlib.pyplot as plt
    
    sh1, sh2, poses = [], [], []
    
    # plt.figure(1)
    for i in range(x1.shape[0]):
        
        x1_tmp = x1[i,0]
        x2_tmp = x2[i,0]
        p_tmp = pos[i]
        
        scale = 1
        pos1 = np.array(( int(p_tmp[1]*scale), int(p_tmp[2]*scale) ))    # y,x: seems correct
        pos2 = np.array(( int(p_tmp[1]*scale), int(p_tmp[0]*scale) )) 
        # print (pos1, pos2)
        # pos1 = np.array([0,32]) # y,x
        # pos2 = np.array([64,64])
        
        # x1_tmp = ob_slice(x1_tmp, pos1-2**(p-1), pos1+2**(p-1)+1)   # 33x33

        # x2_tmp = ob_slice(x2_tmp, pos2-2**(p-1), pos2+2**(p-1)+1)   # 33x33
        x1_tmp = _load_patch_s(x1_tmp, p, pos1)
        x2_tmp = _load_patch_s(x2_tmp, p, pos2)

        # print (x1_tmp.shape)
        # print (x1_tmp.shape)
        sh1.append(x1_tmp)
        sh2.append(x2_tmp)
        
        # d1_tmp = torch.sum(patch_den[i,0],0).numpy()
        # d2_tmp = torch.sum(patch_den[i,0].permute(2,1,0),0).numpy()
        
        # # # d1_tmp = ob_slice(d1_tmp, pos1-2**(p-1), pos1+2**(p-1)+1)
        # # # d2_tmp = ob_slice(d2_tmp, pos2-2**(p-1), pos2+2**(p-1)+1)
        
        # # check patch and sketch correspondence
        # plt.figure(1)
        # plt.subplot(221)
        # plt.imshow(1-x1_tmp[0], cmap='gray')
        # # plt.subplot(222)
        # # plt.imshow(x1_tmp[1], cmap='gray')
        # # plt.subplot(223)
        # # plt.imshow(x1_tmp[2], cmap='gray')
        # # plt.subplot(224)
        # # plt.imshow(x1_tmp[3], cmap='gray')
        # plt.subplot(222)
        # plt.imshow(1-x2_tmp[0], cmap='gray')
        # plt.subplot(223)
        # plt.imshow(d1_tmp, cmap='gray')
        # plt.subplot(224)
        # plt.imshow(d2_tmp, cmap='gray')
        # plt.show()

        # plt.figure(1)
        # plt.subplot(221)
        # plt.imshow(1-x1[i,0], cmap='gray')
        # plt.subplot(222)
        # plt.imshow(1-x2[i,0], cmap='gray')
        # plt.subplot(223)
        # plt.imshow(torch.sum(patch_den[i,0],0), cmap='gray')
        # plt.subplot(224)
        # plt.imshow(torch.sum(patch_den[i,0].permute(2,1,0),0), cmap='gray')
        # plt.show()

    sh1 = np.stack(sh1, 0)
    sh2 = np.stack(sh2, 0)

    # print (s_front.shape, s_side.shape)

    sh1 = torch.FloatTensor(sh1)
    sh2 = torch.FloatTensor(sh2)
    
    # patch_x = np.stack((pu, pv, pw), 0)
    # print (patch_x.shape)

    # plt.figure(1)
    # for i in range(3):
    #     plt.subplot(2,2,i+1)
    #     tmp = patch_x[i,...]
    #     plt.imshow(np.mean(tmp, 0), cmap=plt.cm.RdBu)
    #     print (tmp.shape, tmp.min(), tmp.max())
    # plt.show()
    
    # sh1 = 1 - sh1
    # sh2 = 1 - sh2
    # sh1 = 2 * sh1 - 1
    # sh2 = 2 * sh2 - 1
    # print (sh1.min(), sh1.max(), sh2.min(), sh2.max())
    return sh1, sh2



# %%
