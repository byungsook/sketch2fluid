import argparse
import os
import matplotlib
import numpy as np
import random
import torch
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # 0
if DEVICE.type != 'cpu':
    matplotlib.use('agg')

np.random.seed(42) # cpu vars
random.seed(42) # Python
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42) # gpu vars

import matplotlib.pyplot as plt
from time import time
import datetime
from tqdm import tqdm
import platform

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='synthetic', help='directory containing the data')
parser.add_argument('--arch', type=str, default='ae_d', help='architecture')
parser.add_argument('--data_format', type=str, default='npz', help='npz or vdb')

parser.add_argument('--outd', default='Results', help='directory to save results')
parser.add_argument('--outf', default='Images', help='folder to save synthetic images')
parser.add_argument('--outl', default='Losses', help='folder to save Losses')
parser.add_argument('--outm', default='Models', help='folder to save models')

parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
parser.add_argument('--batch_size', type=int, default=1, help='list of batch sizes during the training')
parser.add_argument('--BN', type=str2bool, default=True, help='use BatchNorm in G and D')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam optimizer')

parser.add_argument('--WS', action='store_true', help='use WeightScale in G and D')
parser.add_argument('--PN', action='store_true', help='use PixelNorm in G')
parser.add_argument('--lrsh', action='store_true', help='use lr scheduler')
parser.add_argument('--test', action='store_true', help='use test mode')
parser.add_argument('--activ_d', type=str, default='tanh', help='if use sigmoid at the end of Generator_d') # tanh

parser.add_argument('--n_iter', type=int, default=1, help='number of epochs to train before changing the progress')
parser.add_argument('--lambdaGP', type=float, default=10, help='lambda for gradient penalty')
parser.add_argument('--gamma', type=float, default=1, help='gamma for gradient penalty')
parser.add_argument('--e_drift', type=float, default=0.001, help='epsilon drift for discriminator loss')
parser.add_argument('--saveimages', type=int, default=1, help='number of epochs between saving image examples')
parser.add_argument('--savenum', type=int, default=64, help='number of examples images to save')
parser.add_argument('--savemodel', type=int, default=10, help='number of epochs between saving models')
parser.add_argument('--savemaxsize', action='store_true', help='save sample images at max resolution instead of real resolution')

# hx: added
parser.add_argument('--output_res', type=int, default=64, help='output resolution')
parser.add_argument('--local_test', action='store_true', help='if local test for plots')
parser.add_argument('--resume', type=str, default=None, help='if local test for plots')
parser.add_argument('--start_epoch', type=int, default=10, help='starting epoch num, controlling the progressive resolution')
parser.add_argument('--MAX_RES', type=int, default=3, help='max resolution of dataset, 3->32, 4->64, 5->128...')
parser.add_argument('--max_epochs', type=int, default=2000, help='max. # epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--resume_epoch', type=int, default=2000, help='resume epoch')
parser.add_argument('--resume_iter', type=int, default=2000, help='resume iter')

parser.add_argument('--w_grad', type=float, default=30, help='weight of gradient loss')
parser.add_argument('--w_sketch', type=float, default=0.02, help='weight of sketch loss')
parser.add_argument('--w_inv_s', type=float, default=0.0, help='weight of sketch loss')
parser.add_argument('--w_proj', type=float, default=0.02, help='weight of projection loss')
parser.add_argument('--w_d', type=float, default=1, help='weight of 3d density loss')
parser.add_argument('--w_tv', type=float, default=0, help='weight of 3d density total variation loss')
parser.add_argument('--w_reg', type=float, default=0, help='weight for regularizer term')
parser.add_argument('--w_render', type=float, default=1, help='weight for regularizer term')
parser.add_argument('--w_prev', type=int, default=1, help='weight for previous one sketch loss')
parser.add_argument('--w_cls', type=int, default=0, help='mask loss')

# seems useless
parser.add_argument('--zdim', type=int, default=256, help='learning rate')
parser.add_argument('--max_fm', type=int, default=512, help='learning rate')
parser.add_argument('--max_i', type=int, default=2000, help='learning rate')
parser.add_argument('--window_size', type=int, default=4, help='window size for second stage')


parser.add_argument('--num_conv', type=int, default=3, help='use physics-based augmentation')
parser.add_argument('--scale', type=int, default=2, help='use physics-based augmentation')
parser.add_argument('--stage', type=str, default='third', help='specify training state: first or second')
parser.add_argument('--log_freq', type=int, default=200, help='log frequency for images and losses')
parser.add_argument('--droprate', type=float, default=0.0, help='dropout rate')
parser.add_argument('--ratio', type=float, default=1, help='dropout rate')
parser.add_argument('--ft_steps', type=int, default=1, help='finetune steps')
parser.add_argument('--refine_seq', type=str, default='fltdrb', help='finetune steps') # fltdrb
parser.add_argument('--val_set', type=str, default='test', help='validation set') # fltdrb

parser.add_argument('--lsm_view', type=int, default=2, help='number of views used in lsm')
parser.add_argument('--lsm_scale', type=float, default=1, help='number of views used in lsm')
parser.add_argument('--lsm_preprocess', type=str, default='hole', help='preprocess way for sketches')
parser.add_argument('--opt_mode', type=str, default='residual', help='preprocess way for sketches')


parser.add_argument('--hks', type=int, default=3, help='half kernel size for gaussian filter in sketcher')
parser.add_argument('--up_size', type=int, default=129, help='upsampling size for density before sketcher')
parser.add_argument('--passes', type=int, default=2, help='number of views used in lsm')
parser.add_argument('--max_passes', type=int, default=5, help='number of views used in lsm')

parser.add_argument('--use_sphere_mask', type=str2bool, default=False, help='if use sphere mask')
parser.add_argument('--use_vel', type=str2bool, default=False, help='if use sphere mask')
parser.add_argument('--use_optim', type=str2bool, default=False, help='if use sphere mask')
parser.add_argument('--use_aug', type=str2bool, default=True, help='if use sphere mask')

parser.add_argument('--run_all_views', type=str2bool, default=False, help='if use sphere mask')

parser.add_argument('--render_scale', type=int, default=1, help='if use sphere mask')
parser.add_argument('--sigmas', type=float, default=0.6, help='if use sphere mask')
parser.add_argument('--milestones', type=int, default=500, help='if use sphere mask')

# --save_npz_pat=/nfs/nas12.ethz.ch/fs1202/infk_ivc_mlfluids/projects/sketch2fluids/results/sketch2den/render/
parser.add_argument('--save_npz_path', type=str, default=None, help='absolute path for saving reconstruction npz data')

opt = parser.parse_args()
      


def main():

    '''
        Set device
    '''
    opt.device = DEVICE
    opt.ngpus = torch.cuda.device_count() # opt.ngpus = 1
    opt.batchSizes = [opt.batch_size]*(opt.MAX_RES+1)   # total batch size for all gpus
    time_stamp = datetime.datetime.now().strftime("%y_%m_%d_%H_%M")

    '''
        Set paths
    '''
    if DEVICE.type != 'cpu':
        opt.workers = 8  # empirical
        opt.data_path = '/net/nas12.ethz.ch/fs1202/infk_ivc_mlfluids/dataset/' # set path here: e.g., '/cluster/scratch/hx/dataset/'    
        opt.results_path = '/net/nas12.ethz.ch/fs1202/infk_ivc_mlfluids/projects/sketch2fluids/hx/'
        opt.outd = opt.results_path + time_stamp + '_' + opt.outd
    else:
        opt.results_path = '../results/'
        opt.data_path = '../'
        opt.outd = opt.results_path + time_stamp + '_' + opt.outd


    if opt.ratio >= 0.1:
        # print ('cudnn = false')
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False
    else:
        print ('cudnn = true')
        torch.backends.cudnn.benchmark = True
    
    '''
        Set Trainer
    '''
    from trainers.hourglass_multiview import Trainer
    if opt.data == 'scalarflow':
        from dataloaders.ScalarflowLoader import Loader
    else:
        from dataloaders.PatchDenLoader import Loader
    
    '''
        Print Config
    '''
    print ('===> Dataset: ', opt.data)
    opt.data = opt.data_path + opt.data
    opt.dmax = 1.4354
    opt.v_scale = 129 # 128
    opt.vmax = 3.2429 #* opt.v_scale # 2.4203 changed here !!!!!!!!!!!!!!!!
    opt.dt = 0.0416667
    opt.vmax_largedt = 2709

    #======================= save arguments ==========================#
    print(opt)
    if not opt.test:
        if not os.path.exists(opt.outd):
            os.makedirs(opt.outd)
        for f in [opt.outf, opt.outl, opt.outm]:
            if not os.path.exists(os.path.join(opt.outd, f)):
                os.makedirs(os.path.join(opt.outd, f))

        opt_file = os.path.join(opt.outd, 'opt.txt')
        with open(opt_file, 'w') as f:
            for k, v in vars(opt).items():
                f.write('%s: %s\n' % (k, v))
           
    '''
        Start Training
    '''
    Trainer(opt, Loader)

if __name__ == '__main__':
    main()