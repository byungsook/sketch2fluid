import argparse
import os
import numpy as np
import torch
from PIL import Image, ImageOps
from scipy import ndimage, misc, stats
import glob

from spatial_transform_rotation import spatial_transformer

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 0

from time import time
from datetime import datetime
from models.pix2vox import Pix2VoxPair_sm, Pix2VoxPairVel
from models.lsm import LSM
import utils
import patch_sketcher


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def load_data(path):
    path = os.path.normpath(path)
    f = glob.glob(path + "\\*")  # all images
    f.sort()
    return f


def _preprocess_single_sketch(path, sigma=0.1, res=258, interp=Image.LANCZOS):
    # deprecated scipy 1.0 code
    # s = Image.open(path).convert('L')
    # s = np.asarray(s)
    # s = misc.imresize(s, (res, res), interp=interp, mode=None) / 255.0
    # s = np.array(s)

    # use this instead
    s = Image.open(path).convert('L').resize(size=(res, res), resample=interp)
    s = np.array(s) / 255.0

    s = ndimage.gaussian_filter(s, sigma=sigma)  ####################### TBD
    s = torch.FloatTensor(s).unsqueeze(0).unsqueeze(0)
    s = torch.flip(s, [2])
    return s

def get_time(unit_min=2):
    output = datetime.now().strftime("%m%d_%H%M")
    minute = (int(output[-2:]) // unit_min) * unit_min
    return output[:-2] + f"{minute:02d}"

pix2vox_pair = None
lsm = None


def sketch2fluid(
        data_front, data_side, output_dir, frame=0, out_frame=0,
        sigma=0.6, bn=True, resume='checkpoint', refine_seq='fl', np_file=None, res=None, rot=[0, 0, 0], velocity=False,
        unit_min=2,
):
    """
        Processes a single frame and stores the result in mapped memory
    """
    data_front = os.path.normpath(data_front.decode('utf-8'))
    data_side = os.path.normpath(data_side.decode('utf-8'))
    output_dir = os.path.normpath(output_dir.decode('utf-8'))

    global pix2vox_pair
    global lsm

    if lsm is None:
        lsm = LSM(device=DEVICE).to(DEVICE)
        pix2vox_pair = Pix2VoxPair_sm(bn=bn).to(DEVICE)

        print('===> Resume checkpoints ...')
        if DEVICE.type == 'cpu':
            pix2vox_pair.load_state_dict(
                torch.load(resume + '/pix2vox_pair_iter-2000.pth', map_location={'cuda:0': 'cpu'}))
        else:
            pix2vox_pair.load_state_dict(torch.load(resume + '/pix2vox_pair_iter-2000.pth'))

    front_list = load_data(data_front)
    side_list = load_data(data_side)

    save_path = ['recon_npz']
    for s in save_path:
        if not os.path.exists(output_dir + '/' + s):
            os.makedirs(output_dir + '/' + s)

    t1 = time()

    f = front_list[frame]
    s = side_list[frame]
    try:
        s_front = _preprocess_single_sketch(f, sigma=sigma).to(DEVICE)
        s_side = _preprocess_single_sketch(s, sigma=sigma).to(DEVICE)
    except:
        print('Could not open frame')
        return None, None

    if np_file is None:
        init_d = lsm([s_front, s_side])
        gen_den_rot = init_d
    else:
        gen_den_rot = np.empty((1, 1, res[0], res[1], res[2]))
        vol = np.memmap(np_file, dtype='float32', mode='r+', shape=res)
        gen_den_rot[0][0] = vol
        gen_den_rot = torch.from_numpy(gen_den_rot).float().to(DEVICE)
        vol._mmap.close()

    ###
    for v in refine_seq:
        if v == 'f':
            sketch_rot = s_front
            view = 'zp'
        if v == 'l':
            sketch_rot = s_side
            view = 'xn'
        transformer = spatial_transformer.SpatialTransformer(DEVICE, gen_den_rot.shape)
        gen_den_rot = transformer(tensor=gen_den_rot, roll=rot[2], pitch=rot[0], yaw=rot[1])
        gen_den_rot = patch_sketcher.rotate(gen_den_rot, view)
        gen_den_rot, _ = pix2vox_pair(gen_den_rot, sketch_rot)
        gen_den_rot = patch_sketcher.rotate_back(gen_den_rot, view)
        transformer = spatial_transformer.SpatialTransformer(DEVICE, gen_den_rot.shape)
        gen_den_rot = transformer(tensor=gen_den_rot, roll=-rot[2], pitch=-rot[0], yaw=-rot[1], order='ypr')

    print('Time to generate: {:.4f}'.format(time() - t1))
    print('rotation is: ' + str(rot))
    print("side shape is " + str(gen_den_rot.shape))
    # np_file = output_dir + '\\' + save_path[0] + '\\' + str(out_frame) + '.npz'
    time_stamp = get_time(unit_min)
    np_file = output_dir + '\\' + save_path[0] + '\\' + f'{out_frame}_{time_stamp}.npz'

    fp = np.memmap(np_file, dtype='float32', mode='w+',
                   shape=(gen_den_rot.size(2), gen_den_rot.size(3), gen_den_rot.size(4)))
    fp[:] = gen_den_rot[0][0].detach().cpu().numpy()[:]  # TODO flush to disk on separate thread?

    # fp = gen_den_rot[0][0].detach().cpu().numpy()
    # np.savez_compressed(np_file, density=fp)

    t2 = time()
    print('Elapsed time: {:.4f}'.format(t2 - t1))

    return np_file, fp.shape


def create_sketches(data_front, data_side, np_file, res, frame=0, rot=[0, 0, 0]):

    """ Create sketches from input density using the differentiable sketcher
        data_front:     saving directory for front sketches
        data_side:      saving directory for side sketches
        np_file:        memory mapped numpy file containing input density
        res:            resolution of input density
        frame:          frame number (number it is saved with)
        rotation:       rotation for interactive rotation in houdini (saved density does not rotate only density in houdini and sketches rotate) """

    data_front = os.path.normpath(data_front.decode('utf-8'))
    data_side = os.path.normpath(data_side.decode('utf-8'))

    gen_den_rot = np.empty((1, 1, res[0], res[1], res[2]))
    vol = np.memmap(np_file, dtype='float32', mode='r+', shape=res)
    gen_den_rot[0][0] = vol
    gen_den_rot = torch.from_numpy(gen_den_rot).float().to(DEVICE)

    transformer = spatial_transformer.SpatialTransformer(DEVICE, gen_den_rot.shape)
    rotated = transformer(tensor=gen_den_rot, roll=rot[2], pitch=rot[0], yaw=rot[1])

    # vol._mmap.close()
    d = torch.clamp(rotated, 0, 1) * 2 - 1

    patch_sketch_render = patch_sketcher.DiffPatchSketchRender(dirs=['zp', 'xn'], upsample_to=256).to(DEVICE)
    patch_sketch_render = torch.nn.DataParallel(patch_sketch_render)

    for param in patch_sketch_render.parameters():
        param.requires_grad = False
    patch_sketch_render.eval()

    ss = patch_sketch_render(d)

    front = (ss[0][3][0][0].detach().cpu().numpy() + 1) * 255 / 2
    front = ImageOps.flip(
        Image.fromarray(front.astype(np.uint8)))
    front.save(data_front + '\\{}.png'.format(str(frame).zfill(2)))

    side = (ss[1][3][0][0].detach().cpu().numpy() + 1) * 255 / 2
    side = ImageOps.flip(Image.fromarray(side.astype(np.uint8)))
    side.save(data_side + '\\{}.png'.format(str(frame).zfill(2)))
