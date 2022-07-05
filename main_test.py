import argparse
import os
import matplotlib
import numpy as np
import random
import torch
from PIL import Image
from scipy import ndimage, misc, stats
import glob
from time import time

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # 0
if DEVICE.type != 'cpu':
    matplotlib.use('agg')


import matplotlib.pyplot as plt
from time import time
import datetime
from tqdm import tqdm
import platform
from models.pix2vox import Pix2VoxPair_sm
from models.lsm import LSM
import render
import utils
import patch_sketcher

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='test/data/dissolve2', help='directory containing the data')
parser.add_argument('--front', type=str, default='FrontF0006.png', help='front view')
parser.add_argument('--side', type=str, default='SideF0006.png', help='side view')
parser.add_argument('--sigma', type=float, default=0.6, help='smooth')
parser.add_argument('--BN', type=str2bool, default=True, help='use BatchNorm in G and D')
parser.add_argument('--resume', type=str, default='checkpoint', help='resumed model')
parser.add_argument('--refine_seq', type=str, default='fl', help='resumed model')

opt = parser.parse_args()
      

def load_data(path):
    f = glob.glob(path+'/*') # all images
    f.sort()
    return f

def _preprocess_single_sketch(path, sigma=0.1, res=258, interp='lanczos'):
    s = Image.open(path).convert('L')
    s = np.asarray(s)
    s = misc.imresize(s, (res,res), interp=interp, mode=None) / 255.0
    s = ndimage.gaussian_filter(s, sigma=sigma) ####################### TBD
    s = torch.FloatTensor(s).unsqueeze(0).unsqueeze(0)
    s = torch.flip(s,[2])
    return s

def main():

    '''
        Set Models
    '''
    opt.device = DEVICE
    lsm = LSM(device=opt.device).to(opt.device)
    pix2vox_pair = Pix2VoxPair_sm(bn=opt.BN).to(opt.device)
    renderer = render.HDRenderer(device=opt.device, resolution = 258, perspective = 1).to(opt.device)

    print ('===> Resume checkpoints ...')
    if DEVICE.type == 'cpu':
        pix2vox_pair.load_state_dict(torch.load(opt.resume+'/pix2vox_pair_iter-2000.pth',map_location={'cuda:0': 'cpu'}))
    else:
        pix2vox_pair.load_state_dict(torch.load(opt.resume+'/pix2vox_pair_iter-2000.pth'))
    pix2vox_pair.eval()

    front_list = load_data(opt.data+'/front')
    side_list = load_data(opt.data+'/side')

    save_path = ['front_recon_render','side_recon_render','front_recon_npz','side_recon_npz']
    for s in save_path:
        if not os.path.exists(opt.data+'/'+s):
            os.makedirs(opt.data+'/'+s)

    t1 = time()

    for f, s in zip(front_list, side_list):
        s_front = _preprocess_single_sketch(f, sigma=opt.sigma)
        s_side  = _preprocess_single_sketch(s, sigma=opt.sigma)
        init_d = lsm([s_front, s_side])
        gen_den_rot = init_d

        ### 
        for v in opt.refine_seq:
            if v == 'f':
                sketch_rot = s_front
                view = 'zp'
            if v == 'l':
                sketch_rot = s_side
                view = 'xn'
        
            gen_den_rot = patch_sketcher.rotate(gen_den_rot, view)
            gen_den_rot, _ = pix2vox_pair(gen_den_rot, sketch_rot)
            gen_den_rot = patch_sketcher.rotate_back(gen_den_rot, view)

        utils.save_render(renderer, gen_den_rot, opt.data+'/'+save_path[0]+'/'+f.split('/')[-1].split('.')[0]+'.png')
        utils.save_render(renderer, patch_sketcher.rotate(gen_den_rot, 'xn'), opt.data+'/'+save_path[1]+'/'+s.split('/')[-1].split('.')[0]+'.png')
        np.savez_compressed(opt.data+'/'+save_path[2]+'/'+f.split('/')[-1].split('.')[0]+'.npz', density=gen_den_rot[0,0].detach().cpu().numpy())
        np.savez_compressed(opt.data+'/'+save_path[3]+'/'+s.split('/')[-1].split('.')[0]+'.npz', density=patch_sketcher.rotate(gen_den_rot, 'xn')[0,0].detach().cpu().numpy())
    
    t2 = time()
    print ('Elapsed time: {:.4f}'.format(t2 - t1))

    
    
        
    
if __name__ == '__main__':
    main()




