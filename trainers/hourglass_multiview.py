import argparse
import os
import matplotlib
import torch
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if DEVICE.type != 'cpu':
    matplotlib.use('agg')

# torch
import matplotlib.pyplot as plt
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.backends import cudnn
import torch.nn.functional as F
# helpers
from time import time
from tensorboardX import SummaryWriter
from tqdm import tqdm
import datetime
from PIL import Image
from collections import defaultdict
import platform
import pickle
# ours
import losses
import patch
import patch_sketcher
import utils
import math
import sketch_jitter
import render
from spatial_transform_rotation.spatial_transformer import SpatialTransformer
from models.pix2vox import Pix2VoxPairVel, Pix2VoxPairOptim, Pix2VoxPair_sm, OptimParams
from models.lsm import LSM
import random


np.random.seed(42)
random.seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)


class Trainer(object):
    def __init__(self, opt, loader):

        '''
            Input: opt, dataloader
            Func: init parameters for training/logging
        '''
        self.fps = 20
        self.steps = 1
        if not opt.test:
            self.summary_writer = SummaryWriter( os.path.join(opt.outd, opt.outl) )
            self.fps = 2
            self.steps = 20

        self.opt = opt
        self.loader = loader
        self.use_sphere_mask = opt.use_sphere_mask
        
        self.train_itr = 0
        self.test_itr = 0

        if self.use_sphere_mask:
            self.large_size = 129
        else:
            self.large_size = 129

        self.load_data(opt)
        self.build_model(opt)

        if opt.test:
            self.test_synthetic(opt, opt.results_path+opt.resume)
            # self.test_artist(opt, opt.results_path+opt.resume)
            # self.test(opt, opt.results_path+opt.resume)

        elif opt.use_optim:
            self.optimization(opt)
        else:
            self.train(opt)

    def load_data(self, opt):
        '''
            Input: opt
            Func: load data to loaders
        '''
        self.train_dset = self.loader(opt.data, 'train', opt.vmax, opt.dmax, opt.max_i, opt.activ_d, opt.window_size, opt.ratio)
        self.train_loader = DataLoader(self.train_dset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers, pin_memory=True, drop_last=True)# worker_init_fn=worker_init_fn)

        self.test_dset = self.loader(opt.data, opt.val_set, opt.vmax, opt.dmax, opt.max_i, opt.activ_d, opt.window_size, opt.ratio)
        self.test_loader = DataLoader(self.test_dset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers, pin_memory=True, drop_last=False)#, worker_init_fn=worker_init_fn)
        
        self.artist_loader = {}
        from dataloaders.TestArtist import Loader
        # for scene, sigma in zip(['dissolve2', 'fraser_iteration3', 'artists'], [0.6, 1.3, 0.6]): # full test
        for scene, sigma in zip(['artists'], [0.6]): # small test
            test_dset = Loader('scenes/'+scene+'/frames', opt.window_size, sigma)
            test_loader = DataLoader(test_dset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
            self.artist_loader[scene] = test_loader

        self.synthetic_loader = {}
        from dataloaders.TestSynthetic import Loader
        # for scene in ['character2', 'character2_cloud', 'smoke_gun', 'turbulent', 'animated_cloud', 'wdas_cloud', 'puppy', 'puppy_cloud', 'sim_000000', 'sim_000100']: # full test
        for scene in ['character2']: # small test
            test_dset = Loader('scenes/'+scene, opt.window_size, self.steps)
            test_loader = DataLoader(test_dset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)
            self.synthetic_loader[scene] = test_loader

        print ( '===> Data successfully loaded: Train {:}, Test {:}'.format(len(self.train_dset),len(self.test_dset)) )
        


    def build_model(self, opt):

        '''
            Input: opt
            Func: build models and tools for training
        '''

        self.opt_params = OptimParams(bn=opt.BN, zdim=opt.zdim, droprate=opt.droprate, nout=1).to(opt.device)
        self.pix2vox_pair = Pix2VoxPair_sm(bn=opt.BN, zdim=opt.zdim, droprate=opt.droprate, nout=1).to(opt.device)
        self.lsm = LSM(device=opt.device, preprocess=opt.lsm_preprocess).to(opt.device)
        print ('===> Number of Parameters Pix2VoxPair: {:.4f} M'.format(sum(p.numel() for p in self.pix2vox_pair.parameters() if p.requires_grad)/1e6 ))

        self.dirs = ['zp', 'xn']
        self.ortho_views = ['zp', 'zn', 'xp', 'xn', 'yp', 'yn']
        self.all_rot = [
            'zp', 'zp1', 'zp2', 'zp3',
            'zn', 'zn1', 'zn2', 'zn3',
            'xp', 'xp1', 'xp2', 'xp3',
            'xn', 'xn1', 'xn2', 'xn3',
            'yp', 'yp1', 'yp2', 'yp3',
            'yn', 'yn1', 'yn2', 'yn3',
        ]

        self.patch_sketch_render_dx1 = patch_sketcher.DiffPatchSketchRender(dirs=self.dirs, kernel_half_size=opt.hks, dx=1).to(opt.device)
        self.patch_sketch_render_dx1_zp = patch_sketcher.DiffPatchSketchRender(dirs=['zp'], kernel_half_size=opt.hks, dx=1).to(opt.device)
        self.patch_sketch_render_6v = patch_sketcher.DiffPatchSketchRender(dirs=self.ortho_views, kernel_half_size=opt.hks, dx=1).to(opt.device)
        
        self.spatial_transformer = SpatialTransformer(device=opt.device, tensor_shape=[1, 1, self.large_size, self.large_size, self.large_size]).to(opt.device)
        
        # Renderer, RendererFaster
        if opt.test:
            self.renderer = render.HDRenderer(device=opt.device, resolution = 258, perspective = 1).to(opt.device)
        else:
            self.renderer = render.RendererFaster(device=opt.device, scale_factor=opt.render_scale, batch_size=opt.batch_size, shape=[1, 1, self.large_size, self.large_size, self.large_size]).to(opt.device)

        self.mask = utils.init_sphere_mask(device=opt.device, res=129, r=64) # [64,91]
        self.patch_sketch_jitter = sketch_jitter.SketchJitter(device=self.opt.device).to(opt.device)

        if opt.resume:
            print ('===> Resume checkpoints ...')
            if DEVICE.type == 'cpu':
                folder = '../results/'+ opt.resume
                self.pix2vox_pair.load_state_dict(torch.load(folder+'/Models/pix2vox_pair_iter-'+str(opt.max_epochs)+'.pth',map_location={'cuda:0': 'cpu'}))
            else:
                folder = '/net/nas12.ethz.ch/fs1202/infk_ivc_mlfluids/projects/sketch2fluids/hx/' + opt.resume
                self.pix2vox_pair.load_state_dict(torch.load(folder+'/Models/pix2vox_pair_iter-'+str(opt.max_epochs)+'.pth'))
                
                
        if opt.ngpus > 1:
            print ('===> multiple gpus')
            self.pix2vox_pair = torch.nn.DataParallel(self.pix2vox_pair)
            self.lsm = torch.nn.DataParallel(self.lsm)
            self.spatial_transformer = torch.nn.DataParallel(self.spatial_transformer)
            self.patch_sketch_render_dx1 = torch.nn.DataParallel(self.patch_sketch_render_dx1)
            self.patch_sketch_render_dx1_zp = torch.nn.DataParallel(self.patch_sketch_render_dx1_zp)
            self.patch_sketch_render_6v = torch.nn.DataParallel(self.patch_sketch_render_6v)
            

        for param in self.patch_sketch_render_dx1.parameters():   
            param.requires_grad = False
        self.patch_sketch_render_dx1.eval()

        for param in self.patch_sketch_render_dx1_zp.parameters():   
            param.requires_grad = False
        self.patch_sketch_render_dx1_zp.eval()
        self.optimizer_pix2vox_pair = Adam(self.pix2vox_pair.parameters(), lr=opt.lr, betas=(0.9, 0.999))
        self.test_no_source = list(range(985*20, 985*20+20)) + list(range(995*20, 995*20+20))
        self.test_w_source = list(range(664*30, 664*30+30)) + list(range(665*30, 665*30+30))
        self.train_no_source = list(range(0*20, 0*20+20)) + list(range(1*20, 1*20+20)) + list(range(2*20, 2*20+20))
        self.train_w_source = list(range(0*30, 0*30+30)) + list(range(1*30, 1*30+30)) + list(range(2*30, 2*30+30))

    def _selected_sim(self, idx, source_type):
        idx = int(idx)
        if self.opt.val_set == 'train':
            if source_type == 'no_source':
                if idx in self.train_no_source:
                    return 1
            if source_type == 'w_source':
                if idx in self.train_w_source:
                    return 1
        elif self.opt.val_set == 'test':
            if source_type == 'no_source':
                if idx in self.test_no_source:
                    return 1
            if source_type == 'w_source':
                if idx in self.test_w_source:
                    return 1
        return 0
                

    def test(self, opt, folder, epoch=0):

        '''
            Run testing on test set of our dataset
        '''
        self.pix2vox_pair.eval()

        f = ['init']
        cur_s = ''
        for s in opt.refine_seq:
            cur_s += s
            f.append(cur_s)

        time_stamp = datetime.datetime.now().strftime("%y_%m_%d_%H_%M")
        out = os.path.join(folder, 'Outputs', '{:0>6}_{:}_{:}'.format(str(epoch), time_stamp, opt.val_set))
        # if not os.path.exists(out):
        #     os.makedirs(out)

        loss_dict = defaultdict(int)

        test_frames = 0

        splitter = '/'
        if platform.system().lower() == 'windows':
            splitter = '\\'

        all_sum_den = []

        for j, (d, fn) in enumerate(self.test_loader):
            fs = fn[0].split(splitter)
            ind = fs[-1].split('_')[-1].split('.')[0]
            scene = fs[-3]
            if self._selected_sim(ind, scene) == 0:
                continue
            else:
                print ('in', scene, ind)
            
            d = d.to(opt.device)

            if self.use_sphere_mask:
                d = utils.apply_sphere_mask(d, self.mask)
            else:
                pass

            test_frames += d.size(0)

            with torch.no_grad():

                s_zp, s_zn, s_xp, s_xn, s_yp, s_yn = self.render_sketches(d, views='zpznxpxnypyn')
                
                s = time()
                gen_den = self._init_guess([s_zp, s_xn])
                den = d
                t1 = time() - s

                gen_den, _, _ = self.run_inference(d, gen_den, [s_zp, s_zn, s_xp, s_xn, s_yp, s_yn], j, out, opt, loss_dict)
                
                ### to test the amount of losing density
                # all_sum_den.append(sum_den)
                # print (sum_den)

                # with open("sum_den.pkl", "wb") as pickle_out:
                #     pickle.dump(all_sum_den, pickle_out)

                # with open("sum_den.pkl", "rb") as pickle_in:
                #     all_sum_den = pickle.load(pickle_in)

                # print (sum_den, all_sum_den)

                if opt.save_npz_path:
                    save_path = opt.save_npz_path + '/' + opt.val_set + '/' + scene
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    for i in range(len(gen_den)):
                        if i == len(gen_den) - 1:
                            np.savez_compressed(save_path+'/'+f[i]+'_{:0>5}'.format(str(ind)), density=gen_den[i][0,0].cpu().numpy(), gt=d[0,0].cpu().numpy())
                        else:
                            if not os.path.exists(save_path+'/'+f[i]):
                                os.makedirs(save_path+'/'+f[i])
                            np.savez_compressed(save_path+'/'+f[i]+'/'+f[i]+'_{:0>5}'.format(str(ind)), density=gen_den[i][0,0].cpu().numpy(), gt=d[0,0].cpu().numpy())
        

        if not opt.test:
            for key, value in loss_dict.items():
                self.summary_writer.add_scalar('test/'+key, value / test_frames, epoch)
            self.summary_writer.add_scalar('test/test_frames', test_frames, epoch)
        
        # utils.convert_png2mp4(out, self.fps)

    def test_synthetic(self, opt, folder, epoch=0):

        '''
            Run tests on our synthetic scenes
        '''
        test_frames = 0
        loss_dict = defaultdict(int)
        f = ['init']
        cur_s = ''
        for s in opt.refine_seq:
            cur_s += s
            f.append(cur_s)

        for scene, test_loader in self.synthetic_loader.items():

            time_stamp = datetime.datetime.now().strftime("%y_%m_%d_%H_%M")
            out = os.path.join(folder, 'Outputs', '{:0>6}_{:}_{:}'.format(str(epoch), time_stamp, scene))
            if not os.path.exists(out):
                os.makedirs(out)

            scale = opt.scale

            s_losses, d_losses = [], []
            ivm_times = []
            update_times = []

            t3 = time()
            for j, (ss, d) in enumerate(test_loader):
                # print (j)
                if not opt.test:
                    if j % 50 != 0:
                        continue
                else:
                    # pass
                    if (j + 1) not in [40, 220, 230]:
                        continue
                    # if j + 1 != len(test_loader)-20: # 21, -20
                    #     continue

                test_frames += 1

                ss = ss.to(opt.device)
                d = d.to(opt.device)
                if d.size(-1) != 129:
                    d = F.interpolate(d, size=129, mode='trilinear')

                with torch.no_grad():

                    if self.use_sphere_mask:
                        d = utils.apply_sphere_mask(d, self.mask)
                    s_zp, s_zn, s_xp, s_xn, s_yp, s_yn = self.render_sketches(d, views='zpznxpxnypyn')

                    ### for sketch aug test ###
                    # brightness = 42.33*1.5
                    # s_zp_aug = self.patch_sketch_jitter.brightness_contrast(s_zp, brightness, 0)
                    # s_xn_aug = self.patch_sketch_jitter.brightness_contrast(s_xn, brightness, 0)
                    
                    # contrast = 64
                    # s_zp = self.patch_sketch_jitter.brightness_contrast(s_zp, 0, contrast)
                    # s_xn = self.patch_sketch_jitter.brightness_contrast(s_xn, 0, contrast)
                    
                    # sketchJitter = sketch_jitter.SketchJitter(device=opt.device, sigma=1.5)
                    # s_zp = sketchJitter.blur(s_zp)
                    # s_xn = sketchJitter.blur(s_xn)
                    
                    # shift = 0.06
                    # s_zp = self.patch_sketch_jitter.slur(s_zp, [0,shift])
                    # s_xn = self.patch_sketch_jitter.slur(s_xn, [0,shift])

                    # light_dir
                    # toon color
                    # contour
                    
                    utils.save_sketch(s_zp, out+'/gt_s_front_{:0>5}.png'.format(str(j+1)), out_size=1024)
                    self.pix2vox_pair.eval()
                    
                    t1 = time()
                    gen_den = self._init_guess([s_zp, s_xn])
                    t2 = time()
                    ivm_times.append(t2 - t1)

                    gen_den, s_loss, d_loss, t = self.run_inference(d, gen_den, [s_zp, s_zn, s_xp, s_xn, s_yp, s_yn], j+1, out, opt, loss_dict)
                    update_times.append(t)
                    s_losses.append(s_loss)
                    d_losses.append(d_loss)
                    
                    if opt.save_npz_path:
                        save_path = opt.save_npz_path + '/' + scene
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        for i in range(len(gen_den)):
                            if i == len(gen_den) - 1:
                                np.savez_compressed(save_path+'/'+f[i]+'_{:0>5}'.format(str(j+1)), density=gen_den[i][0,0].cpu().numpy())
                            else:
                                if not os.path.exists(save_path+'/'+f[i]):
                                    os.makedirs(save_path+'/'+f[i])
                                np.savez_compressed(save_path+'/'+f[i]+'/'+f[i]+'_{:0>5}'.format(str(j+1)), density=gen_den[i][0,0].cpu().numpy())
            # utils.convert_png2mp4(out, self.fps)
            
            t4 = time()
            n_frames = len(test_loader)
            print ('{:}: {:} frames, total_time {:.4f}s, total_ivm {:.4f}s, avg_ivm {:.4f}s, total_update {:.4f}s, avg_update {:.4f}s, s_loss {:.4f}, d_loss {:.4f}'.format(
                    scene, n_frames, t4-t3, sum(ivm_times), sum(ivm_times)/n_frames, sum(update_times), sum(update_times)/n_frames, np.mean(s_losses), np.mean(d_losses)))
        
        if not opt.test:
            for key, value in loss_dict.items():
                self.summary_writer.add_scalar('test_synthetic/'+key, value/test_frames, epoch)
            self.summary_writer.add_scalar('test_synthetic/test_frames', test_frames, epoch)

    def test_artist(self, opt, folder, epoch=0):
        
        '''
            Run tests on artist sketches
        '''

        refine_seq = 'fl'
        f = ['init']
        cur_s = ''
        for s in refine_seq:
            cur_s += s
            f.append(cur_s)

        for scene, test_loader in self.artist_loader.items():

            time_stamp = datetime.datetime.now().strftime("%y_%m_%d_%H_%M")
            out = os.path.join(folder, 'Outputs', '{:0>6}_{:}_{:}'.format(str(epoch), time_stamp, scene))
            
            if not os.path.exists(out):
                os.makedirs(out)

            for i, (sketches, d) in enumerate(test_loader):
                
                N = sketches.size(1)
                d = d.to(opt.device)
                sketches = sketches.to(opt.device)
                d0 = torch.flip(d,[3]).unsqueeze(1)
                
                # print (sketches.size()) # [1, N, 3, 129, 129]

                for j in range(0,N):
                    
                    # if j + 1 != 13: # small test
                    #     continue
                    s_front, s_side, s_top = sketches[:,j,0:1,...], sketches[:,j,1:2,...], sketches[:,j,2:3,...]
                    s_front_ss = torch.flip(s_front, [2]) # flip for coordinate consistent
                    s_side_ss = torch.flip(s_side, [2])
                    s_top_ss = torch.flip(s_top, [2])

                    if self.use_sphere_mask:
                        pass

                    generated_densities = []

                    with torch.no_grad():

                        self.pix2vox_pair.eval()
                        t1 = time()
                        gen_den = self._init_guess([s_front_ss, s_side_ss])
                        t2 = time()

                        avg_time = t2 - t1
                        print ('IVM time: {:.4f}'.format(avg_time))

                        s_front_recon, s_side_recon = self.render_sketches(gen_den)

                        # utils.save_render(self.renderer, gen_den, out+'/lsm_r_front_{:0>5}.png'.format(str(j+1)))
                        # utils.save_render(self.renderer, patch_sketcher.rotate(gen_den, 'xn'), out+'/lsm_r_side_{:0>5}.png'.format(str(j+1)))
                        # utils.save_sketch(s_front_recon, out+'/lsm_s_front_{:0>5}.png'.format(str(j+1)), out_size=1024) 
                        # utils.save_sketch(s_side_recon, out+'/lsm_s_side_{:0>5}.png'.format(str(j+1)), out_size=1024)

                        utils.save_sketch(s_front_ss, out+'/gt_s_front_{:0>5}.png'.format(str(j+1)), out_size=1024)
                        utils.save_sketch(s_side_ss, out+'/gt_s_side_{:0>5}.png'.format(str(j+1)), out_size=1024)

                        generated_densities.append(gen_den)

                        cur_t = ''
                        for t in refine_seq:
                            cur_t += t
                            if t == 'f':
                                sketch_rot = s_front_ss
                                view = 'zp'
                            elif t == 'l':
                                sketch_rot = s_side_ss
                                view = 'xn'
                            elif t == 't':
                                sketch_rot = s_top_ss
                                view = 'yp'
                            else:
                                raise Exception ("input views should be f / s")
                            gen_den_rot = patch_sketcher.rotate(gen_den, view)

                            t3 = time()
                            gen_den_rot, _ = self.pix2vox_pair(gen_den_rot, sketch_rot)
                            t4 = time()

                            print ('Update time: {:.4f}'.format(t4 - t3))
                            # s_cur = self.render_sketches(gen_den_rot, views='zp')

                            gen_den = patch_sketcher.rotate_back(gen_den_rot, view)

                            # gen_den = self.resample(gen_den, opt)
                            s_front_recon, s_side_recon = self.render_sketches(gen_den)
                            
                            generated_densities.append(gen_den)

                            # utils.save_render(self.renderer, gen_den, out+'/'+cur_t+'_r_front_{:0>5}.png'.format(str(j+1)))
                            # utils.save_render(self.renderer, patch_sketcher.rotate(gen_den, 'xn'), out+'/'+cur_t+'_r_side_{:0>5}.png'.format(str(j+1)))
                            utils.save_sketch(s_front_recon, out+'/'+cur_t+'_s_front_{:0>5}.png'.format(str(j+1)))
                            # utils.save_sketch(s_side_recon, out+'/'+cur_t+'_s_side_{:0>5}.png'.format(str(j+1)))
                            
                            print (
                                '{:}: front sketch loss {:.4f}, side sketch loss {:.4f}'.format(
                                    cur_t,
                                    losses.sketchLoss(s_front_recon, s_front_ss).item(),
                                    losses.sketchLoss(s_side_recon, s_side_ss).item(),
                                )   
                            )

                        if opt.save_npz_path:
                            save_path = opt.save_npz_path + '/' + scene
                            for k in range(len(generated_densities)):
                                if k != len(generated_densities) - 1:
                                    if not os.path.exists(save_path + '/' + f[k]):
                                        os.makedirs(save_path + '/' + f[k])
                                    np.savez_compressed(save_path+'/{:}/{:}_{:0>5}'.format(f[k], f[k], str(j+1)), density=gen_den[0,0].cpu().numpy())
                                else:
                                    np.savez_compressed(save_path+'/{:}_{:0>5}'.format(f[k], str(j+1)), density=gen_den[0,0].cpu().numpy())
                                    
            # utils.convert_png2mp4(out, 2)



    def run_inference(self, d, gen_den, s, j, out, opt, loss_dict):

        '''
            Helper function for running synthetic and artist tests
        '''

        run_all_views = self.opt.run_all_views # False
        vs = ['zp', 'xn', 'yp', 'yn', 'xp', 'zn']
        cur_t = ''
        # s_zp, s_zn, s_xp, s_xn, s_yp, s_yn, s_zp_aug, s_xn_aug = s # for augmentation test
        s_zp, s_zn, s_xp, s_xn, s_yp, s_yn = s
        
        prev_v = []
        prev_s = []
        generated_densities = [gen_den]
        sum_densities = [torch.sum(gen_den).item()]
        
        act_ind = j - 1
        times = 0

        s_cur_recon = self.render_sketches(gen_den, views='zp')
        s_prev_recon = self.render_sketches(patch_sketcher.rotate(gen_den, 'xn'), views='zp')
        print (
            '{:}: front sketch loss {:.4f}, side sketch loss {:.4f}, density loss {:.4f}'.format(
                'ivm',
                losses.sketchLoss(s_cur_recon, s_zp).item(),
                losses.sketchLoss(s_prev_recon, s_xn).item(),
                losses.densityLoss3D(gen_den, d).item()
            )   
        )

        # gen_den = d
        dr = d
        if not run_all_views:
            if not opt.test:
                if act_ind % 50 == 0:
                    utils.save_render(self.renderer, gen_den, out+'/lsm_r_front_{:0>5}.png'.format(str(j)))
                    utils.save_render(self.renderer, patch_sketcher.rotate(gen_den, 'xn'), out+'/lsm_r_side_{:0>5}.png'.format(str(j)))
                    utils.save_render(self.renderer, d, out+'/gt_r_front_{:0>5}_.png'.format(str(j)))
                    utils.save_render(self.renderer, patch_sketcher.rotate(d, 'xn'), out+'/gt_r_side_{:0>5}.png'.format(str(j)))
                    utils.save_sketch(s_zp, out+'/gt_s_front_{:0>5}.png'.format(str(j)))
                    utils.save_sketch(s_xn, out+'/gt_s_side_{:0>5}.png'.format(str(j)))
            else:
                # utils.save_render(self.renderer, gen_den, out+'/lsm_r_front_{:0>5}.png'.format(str(j)))
                # utils.save_render(self.renderer, patch_sketcher.rotate(gen_den, 'xn'), out+'/lsm_r_side_{:0>5}.png'.format(str(j)))
                # utils.save_render(self.renderer, d, out+'/gt_r_front_{:0>5}_.png'.format(str(j)))
                # utils.save_render(self.renderer, patch_sketcher.rotate(d, 'xn'), out+'/gt_r_side_{:0>5}.png'.format(str(j)))
                # utils.save_sketch(s_zp, out+'/gt_s_front_{:0>5}.png'.format(str(j)))
                # utils.save_sketch(s_xn, out+'/gt_s_side_{:0>5}.png'.format(str(j)))
                pass
        else:
            tmp_d = d
            for p in vs:
                tmp_d = patch_sketcher.rotate(tmp_d, p)
                tmp_s = self.render_sketches(tmp_d, views='zp')
                utils.save_sketch(tmp_s, out+'/gt_s_{:}_{:0>5}.png'.format(p, str(j)))
                utils.save_render(self.renderer, tmp_d, out+'/gt_r_{:}_{:0>5}.png'.format(p, str(j)))
                tmp_d = patch_sketcher.rotate_back(tmp_d, p)

        # fltdrb test
        rot_angles_dict = {'f':[0,0,0], 'ff':[0,0,30], 'fff':[0,0,30], 'ffff':[0,30,30], 'fffff':[0,30,30]} # for arbitrary views
        
        for t in opt.refine_seq:
            cur_t += t
            if t == 'f':
                sketch_rot = s_zp; view = 'zp'
            elif t == 'l':
                sketch_rot = s_xn; view = 'xn'
            elif t == 't':
                sketch_rot = s_yp; view = 'yp'
            elif t == 'd':
                sketch_rot = s_yn; view = 'yn'
            elif t == 'r':
                sketch_rot = s_xp; view = 'xp'
            elif t == 'b':
                sketch_rot = s_zn; view = 'zn'
            
            else:
                raise Exception ("input views should be 6 canonical views")

            # canonical views
            gen_den_rot = patch_sketcher.rotate(gen_den, view)

            # arbitrary views
            # gen_den_rot = self.spatial_transformer(gen_den, rot_angles_dict[cur_t][0], rot_angles_dict[cur_t][1], rot_angles_dict[cur_t][2], order='rpy')
            # dr = self.spatial_transformer(dr, rot_angles_dict[cur_t][0], rot_angles_dict[cur_t][1], rot_angles_dict[cur_t][2], order='rpy')
            # sketch_rot = self.render_sketches(dr, views='zp')
            t1 = time()

            gen_den_rot, _ = self.pix2vox_pair(gen_den_rot, sketch_rot)

            t2 = time()
            times += t2 - t1
            # current view sketch loss
            # gen_den_rot = self.resample(gen_den_rot, opt)

            prev_s.append(sketch_rot)
            prev_v.append(view)

            s_cur_recon = self.render_sketches(gen_den_rot, views='zp')
            loss_dict[cur_t+'_s_l1'] += F.l1_loss(s_cur_recon, sketch_rot).item()
            loss_dict[cur_t+'_tv_l1'] += losses.totalVariation3D(gen_den_rot).item()

            # canonical views
            gen_den = patch_sketcher.rotate_back(gen_den_rot, view)

            # arbitrary views
            # gen_den = self.spatial_transformer(gen_den_rot, 0, 0, -rot_angles, order='ypr')
            # gen_den = gen_den_rot

            loss_dict[cur_t+'_d_l1'] += losses.densityLoss3D(gen_den, d).item()

            s_cur_recon = self.render_sketches(gen_den, views='zp')
            gen_den_prev = patch_sketcher.rotate(gen_den, 'xn')
            s_prev_recon = self.render_sketches(gen_den_prev, views='zp')

            # previous all sketch losses
            generated_densities.append(gen_den)
            sum_densities.append(torch.sum(gen_den).item())
            
            # gen_den = torch.clamp(gen_den*1.3,0,1) # compensate density
            tmp_d = gen_den

            if not run_all_views:
                if not opt.test:
                    if act_ind % 50 == 0:
                        utils.save_sketch(s_cur_recon, out+'/'+cur_t+'_s_front_{:0>5}.png'.format(str(j)))
                        utils.save_render(self.renderer, gen_den, out+'/'+cur_t+'_r_front_{:0>5}.png'.format(str(j)))
                        utils.save_sketch(s_prev_recon, out+'/'+cur_t+'_s_side_{:0>5}.png'.format(str(j)))
                        utils.save_render(self.renderer, gen_den_prev, out+'/'+cur_t+'_r_side_{:0>5}.png'.format(str(j)))

                    if len(cur_t) > 1:
                        for p in range(len(prev_v)-2,-1,-1):
                            # print (len(cur_t), p, prev_v[p])
                            tmp_d = patch_sketcher.rotate(tmp_d, prev_v[p])
                            tmp_s = self.render_sketches(tmp_d, views='zp')
                            loss_dict[cur_t+'_s_l1'] += F.l1_loss(tmp_s, prev_s[p]).item()
                            tmp_d = patch_sketcher.rotate_back(tmp_d, prev_v[p])

                else:
                    utils.save_sketch(s_cur_recon, out+'/'+cur_t+'_s_front_{:0>5}.png'.format(str(j)))
                    # utils.save_render(self.renderer, gen_den, out+'/'+cur_t+'_r_front_{:0>5}.png'.format(str(j)))
                    utils.save_sketch(s_prev_recon, out+'/'+cur_t+'_s_side_{:0>5}.png'.format(str(j)))
                    # utils.save_sketch(sketch_rot, out+'/'+cur_t+'_s_rot_{:0>5}.png'.format(str(j)))
                    # utils.save_render(self.renderer, gen_den_prev, out+'/'+cur_t+'_r_side_{:0>5}.png'.format(str(j)))

                    pass
            else:
                for p in vs:
                    # print (len(cur_t), p, prev_v[p])
                    tmp_d = patch_sketcher.rotate(tmp_d, p)
                    tmp_s = self.render_sketches(tmp_d, views='zp')
                    utils.save_sketch(tmp_s, out+'/'+cur_t+'_s_{:}_{:0>5}.png'.format(p, str(j)))
                    utils.save_render(self.renderer, tmp_d, out+'/'+cur_t+'_r_{:}_{:0>5}.png'.format(p, str(j)))
                    tmp_d = patch_sketcher.rotate_back(tmp_d, p)
                

            # utils.plot_histogram(gen_den, out+'/hist_recon_{:0>5}.png'.format(str(i))) # histogram test
            # utils.plot_histogram(d, out+'/hist_gt_{:0>5}.png'.format(str(i)))
            print (
                '{:}: front sketch loss {:.4f}, side sketch loss {:.4f}, density loss {:.4f}'.format(
                    cur_t,
                    losses.sketchLoss(s_cur_recon, s_zp).item(),
                    losses.sketchLoss(s_prev_recon, s_xn).item(),
                    losses.densityLoss3D(gen_den, d).item()
                )   
            )
        s_loss = (losses.sketchLoss(s_cur_recon, s_zp).item() + losses.sketchLoss(s_prev_recon, s_xn).item()) / 2
        d_loss = losses.densityLoss3D(gen_den, d).item()
        
        return generated_densities, s_loss, d_loss, times
        

    def _process_single_sketch(self, ss, view, style):
        '''
            Helper function to process a single view sketch
        '''
        s = ss[view][style]
        s = s * 0.5 + 0.5
        # s = F.interpolate(s, scale_factor=2, mode='bilinear')
        s = F.interpolate(s, size=258, mode='bilinear')
        return s

    def _gen_augment_params(self):
        '''
            Helper functions to generate CONSISTENT random parameters for data augmentation
        '''

        rns = {}

        
        if self.opt.use_aug: # gaussian sampling
            rns['contour'] = 0.8 - np.abs(utils.gaussian_sampling(0,0.1))
            x_dir = 1 + utils.gaussian_sampling(0,0.66)
            y_dir = 1 + utils.gaussian_sampling(0,0.66)
            rns['tooncolor'] = utils.gaussian_sampling(0.8, 0.03) # 0.8,0.03
            rns['blur'] = np.abs(utils.gaussian_sampling(0,0.5)) + 1e-4
            # rns['brightness'] = np.abs(utils.gaussian_sampling(0,42.33))
            # rns['contrast'] = np.abs(utils.gaussian_sampling(0,21.33))
            rns['brightness'] = np.abs(utils.gaussian_sampling(0,42.33))
            rns['contrast'] = np.abs(utils.gaussian_sampling(0,21.33))
            shift = np.abs(utils.gaussian_sampling(0,0.02))
            prob_thresh = 0.2 # 0.2

            # else: # uniform sampling
            #     rns['contour'] = np.random.uniform(0.5,0.8)
            #     x_dir = np.random.uniform(-1.2, 1.2)
            #     y_dir = np.random.uniform(-1.2, 1.2)
            #     rns['tooncolor'] = np.random.uniform(0.7,0.9) # 0.8,0.03
            #     rns['blur'] = np.abs(np.random.uniform(0,1.5))
            #     rns['brightness'] = np.random.uniform(0,120)
            #     rns['contrast'] = np.random.uniform(0,60)
            #     shift = np.abs(utils.gaussian_sampling(0,0.02))

        # no aug
        else:
            rns['contour'] = 0.8
            x_dir = 1
            y_dir = 1
            rns['tooncolor'] = 0.8
            rns['blur'] = 1e-4 # 1e-4
            rns['brightness'] = 0
            rns['contrast'] = 0
            shift = 0
            prob_thresh = 0.0 # 0.0

        ld = [x_dir,y_dir,1]
        denom = math.sqrt( sum( [i*i for i in ld] ) )
        ld_norm = [ld[i] / denom for i in range(len(ld))]
        rns['lightdir'] = ld_norm
        choices = [[0,-shift],[0,shift],[-shift,0],[shift,0]]
        rns['slur'] = choices[np.random.choice(4)]
        
        
        for t in range(self.opt.passes):
            p = np.random.uniform(-180,180,3)
            rns['roll'+str(t+1)] = p[0]
            rns['pitch'+str(t+1)] = p[1]
            rns['yaw'+str(t+1)] = p[2]
            ra = self.all_rot[np.random.choice(24)]
            rns['ortho'+str(t+1)] = ra

        prob = np.random.random()
        if prob < prob_thresh:
            rns['noise'] = True
        else:
            rns['noise'] = False
        return rns

    def render_sketches(self, d, style_index=-1, rns=None, views='zpxn'):
        '''
            return 3-view sketches with size (2, 1, 33, 33) and range (0,1)
            inputs:
                3d density: [0,1]
                style: default = -1
                random numbers: default = None
        '''
        if self.opt.up_size > 129:
            d = F.interpolate(d, size=self.opt.up_size, mode='trilinear')
        # print (d.size())
        d = torch.clamp(d, 0, 1) * 2 - 1
        if rns is None:
            if views == 'zp':
                # print ('1-view')
                ss = self.patch_sketch_render_dx1_zp(d)
                s_front = self._process_single_sketch(ss, 0, style_index)
                return s_front

            elif views == 'zpxn':
                # print ('2-view')
                ss = self.patch_sketch_render_dx1(d)
                s_front = self._process_single_sketch(ss, 0, style_index)
                s_side = self._process_single_sketch(ss, 1, style_index)
                # s_top = self._process_single_sketch(ss, 2, style_index)
                return s_front, s_side

            elif views == 'zpznxpxnypyn':
                # print ('light dir augmentation test')
                # ld = [3,1,1]
                # denom = math.sqrt( sum( [i*i for i in ld] ) )
                # ld_norm = [ld[i] / denom for i in range(len(ld))]

                ss = self.patch_sketch_render_6v(d) # light_dir=ld_norm, toon_color=0.7, contour_thresh=0.5
                s = []
                for i in range(6):
                    s.append(self._process_single_sketch(ss, i, style_index))                
                return s

        if views == 'zp':
            ss = self.patch_sketch_render_dx1_zp(d, contour_thresh=rns['contour'], toon_color=rns['tooncolor'], light_dir=rns['lightdir'])
            s_front = self._process_single_sketch(ss, 0, style_index)
            s_front = self.patch_sketch_jitter.brightness_contrast(s_front, rns['brightness'], rns['contrast'])
            return s_front

        elif views == 'zpxn':
            ss = self.patch_sketch_render_dx1(d, contour_thresh=rns['contour'], toon_color=rns['tooncolor'], light_dir=rns['lightdir'])
            s_front = self._process_single_sketch(ss, 0, style_index)
            s_side = self._process_single_sketch(ss, 1, style_index)
            s_front = self.patch_sketch_jitter.brightness_contrast(s_front, rns['brightness'], rns['contrast'])
            s_side = self.patch_sketch_jitter.brightness_contrast(s_side, rns['brightness'], rns['contrast'])
            return s_front, s_side


    def _augment_sketches(self, s, rns=None):
        '''
            Func: 
                Augment multi-view sketches
            Inputs:
                Sketches from multiple views
            Outputs:
                Jitter sketches
        '''
        if rns is None:
            return s
        out = []
        # Do not support multiple gpus as it seems already fast
        patch_sketch_jitter = sketch_jitter.SketchJitter(device=self.opt.device, sigma=rns['blur']).to(self.opt.device)
        for sketch in s:
            # s_front, s_side, s_top = s
            sketch_aug = patch_sketch_jitter.blur(sketch)
            sketch_aug = patch_sketch_jitter.slur(sketch_aug, rns['slur'])
            out.append(sketch_aug)
        return out
        
    def _resize_test_sketch(self, s):
        small_size = s.size(-1)
        large_size = 384
        assert (large_size - small_size) % 2 == 0
        pad = (large_size - small_size) // 2
        pads = [pad-1, pad+1] * 2
        s = F.pad(s, pad=pads, value=1)
        return s

    def _init_guess(self, ss):
        d = self.lsm(ss, self.opt.lsm_scale)
        return d


    def optimization(self, opt):

        '''
            Run optimization tests for comparisons
        ''' 
        print ('===> Optimization stage {} ...'.format(opt.stage))
        
        opt_synthetic = True

        if not opt_synthetic:
            from dataloaders.TestArtist import Loader
            test_dset = Loader('scenes/'+'dissolve'+'/frames', opt.window_size)
            test_loader = DataLoader(test_dset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
        else:
            from dataloaders.TestSynthetic import Loader
            test_dset = Loader('scenes/smoke_gun', opt.window_size, self.steps)
            test_loader = DataLoader(test_dset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

        optim = Adam(self.opt_params.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        
        for i, (ss, ds) in enumerate(test_loader):

            if opt_synthetic and (i + 1) != 100:
                continue

            ds = ds.to(opt.device); d0 = ds
            ss = ss.to(opt.device)
            augment_params = self._gen_augment_params()

            if opt_synthetic:
                s_front, s_side = self.render_sketches(d0, rns=None)
            else:
                s_front, s_side, s_top = ss[:,0,0:1,...], ss[:,0,1:2,...], ss[:,0,2:3,...]
            den_theta1 = d0
            
            self.pix2vox_pair.eval()
            self.opt_params.train()
            
            mode = opt.opt_mode

            # def closure(): # for lbfgs

            histo = []

            with torch.no_grad():
                lsm = self._init_guess([s_front, s_side])
                # lsm, _ = self.pix2vox_pair(lsm, s_front)

            for j in range(opt.max_epochs):

                t = time()
                optim.zero_grad()

                gen_den_theta1 = self.opt_params(lsm, mode)

                s_front_recon, s_side_recon = self.render_sketches(gen_den_theta1, rns=None)
                

                # s_l1 = losses.sketchLoss(s_front_recon, s_front) #+ losses.sketchLoss(s_side_recon, s_side)) / 2
                s_l1 = losses.sketchLoss(s_side_recon, s_side) + losses.sketchLoss(s_front_recon, s_front)
                
                # inv_s_l1 = patch_sketcher.inverse_sketch_loss(d=gen_den_theta1, dx=1, c=5, d_prev=lsm, alpha=1)
                
                d_l1 = losses.densityLoss3D(gen_den_theta1, den_theta1)
                # grad_l1 = losses.densityGradLoss(gen_den_theta1, den_theta1)
                tv_l1 = losses.totalVariation3D(gen_den_theta1, mode=1) # residual_gen_den_theta1, gen_den_theta1

                # d_loss = opt.w_d * d_l1 + opt.w_sketch * s_l1 + opt.w_tv * tv_l1
                # d_loss = opt.w_d * d_l1 #+ opt.w_sketch * s_l1 + opt.w_inv_s * inv_s_l1
                d_loss = s_l1 #opt.w_sketch * s_l1 + opt.w_tv * tv_l1
                
                # print (s_l1.item(), inv_s_l1.item())
                
                d_loss.backward(retain_graph=True)

                ti = time() - t
                print (ti)

                if self.train_itr % opt.log_freq == 0:

                    utils.save_render(self.renderer, gen_den_theta1, os.path.join(opt.outd, opt.outf, f'{self.train_itr:06d}_p1_r_cur_recon.png'))
                    utils.save_render(self.renderer, patch_sketcher.rotate(gen_den_theta1, 'xn'), os.path.join(opt.outd, opt.outf, f'{self.train_itr:06d}_p1_r_prev_recon.png'))
                    
                    utils.save_sketch(s_front_recon, os.path.join(opt.outd, opt.outf, f'{self.train_itr:06d}_p1_s_cur_recon.png'))
                    utils.save_sketch(s_side_recon, os.path.join(opt.outd, opt.outf, f'{self.train_itr:06d}_p1_s_prev_recon.png'))
                    
                    self.summary_writer.add_scalar('train_p1/s_l1', s_l1.item(), self.train_itr)
                    self.summary_writer.add_scalar('train_p1/d_l1', d_l1.item(), self.train_itr)
                    self.summary_writer.add_scalar('train_p1/tv_l1', tv_l1.item(), self.train_itr)
                    self.summary_writer.add_scalar('train_p1/time', ti, self.train_itr)

                    if j == 0 or j == opt.max_epochs - 1:
                        utils.save_render(self.renderer, den_theta1, os.path.join(opt.outd, opt.outf, f'{self.train_itr:06d}_p1_r_cur_gt.png'))
                        utils.save_render(self.renderer, patch_sketcher.rotate(den_theta1, 'xn'), os.path.join(opt.outd, opt.outf, f'{self.train_itr:06d}_p1_r_prev_gt.png'))
                        utils.save_sketch(s_front, os.path.join(opt.outd, opt.outf, f'{self.train_itr:06d}_p1_s_cur_gt.png'))
                        utils.save_sketch(s_side, os.path.join(opt.outd, opt.outf, f'{self.train_itr:06d}_p1_s_prev_gt.png'))
                    
                        np.savez_compressed(os.path.join(opt.outd, opt.outf, 'out_{:0>5}'.format(str(j))), density=gen_den_theta1[0,0].detach().cpu().numpy())
                    
                histo.append(d_loss.item())
                self.train_itr += 1
                # return d_loss
                optim.step()
            # optim.step(closure)

            return

    def train(self, opt):
        print ('===> Training ...')
        self.summary_writer.add_text('text', opt.stage)

        scale = opt.scale
        epoch = 0

        w_sketch = opt.w_sketch
        w_d = opt.w_d
        w_tv = opt.w_tv
        w_proj = opt.w_proj
        w_render = opt.w_render
        w_grad = opt.w_grad
        w_tv = opt.w_tv
        w_cls = opt.w_cls
        w_prev = 1
        
        passes = self.opt.passes

        while epoch < opt.max_epochs:
            
            t0 = time()

            for i, (ds, fn) in enumerate(self.train_loader):

                if self.train_itr and not self.train_itr % opt.milestones: 
                    w_prev += 10
                    if w_prev >= opt.w_prev:
                        w_prev = opt.w_prev

                ds = ds.to(opt.device); d0 = ds
                augment_params = self._gen_augment_params()

                if self.use_sphere_mask:

                    d0 = utils.apply_sphere_mask(d0, self.mask)
                    d0 = self.spatial_transformer(d0, augment_params['roll1'], augment_params['pitch1'], augment_params['yaw1'], order='rpy')
                    
                else:
                    d0 = patch_sketcher.rotate(d0, augment_params['ortho1'])
                    
                s_front, s_side = self.render_sketches(d0, rns=augment_params)

                den_theta1 = d0
                
                if augment_params is not None and augment_params['noise'] == True:
                    s_front_ss, s_side_ss = self._augment_sketches([s_front, s_side], rns=augment_params)
                else:
                    s_front_ss, s_side_ss = s_front, s_side


                gen_den_theta1 = self._init_guess([s_front_ss, s_side_ss])

                # utils.save_render(self.renderer, gen_den_theta1, os.path.join(opt.outd, opt.outf, f'{self.train_itr:06d}_p0_r_cur_recon.png'))
                
                self.pix2vox_pair.train()
                self.pix2vox_pair.zero_grad()
                
                gen_den_theta1, residual_gen_den_theta1 = self.pix2vox_pair(gen_den_theta1, s_front_ss)
                s1_front_recon, s1_side_recon = self.render_sketches(gen_den_theta1, rns=augment_params)
                # r1_front_recon = self.renderer(gen_den_theta1)
                # r_front = self.renderer(den_theta1)
                
                s_l1 = losses.sketchLoss(s1_front_recon, s_front) #+ losses.sketchLoss(s1_side_recon, s_side)
                # render_l1 = F.l1_loss(r1_front_recon, r_front)
                d_l1 = losses.densityLoss3D(gen_den_theta1, den_theta1)
                cls_loss = losses.clsLoss(gen_den_theta1, s_front)

                # grad_l1 = losses.densityGradLoss(gen_den_theta1, den_theta1)
                tv_l1 = losses.totalVariation3D(gen_den_theta1) # residual_gen_den_theta1, gen_den_theta1
                proj_l1 = losses.densityProjectionLoss(gen_den_theta1, den_theta1)

                if self.train_itr % opt.log_freq == 0:
                    utils.save_render(self.renderer, gen_den_theta1, os.path.join(opt.outd, opt.outf, f'p1_r_front_{self.train_itr:06d}.png'))
                    utils.save_render(self.renderer, den_theta1, os.path.join(opt.outd, opt.outf, f'gt_r_front_{self.train_itr:06d}.png'))
                    utils.save_sketch(s1_front_recon, os.path.join(opt.outd, opt.outf, f'p1_s_front_{self.train_itr:06d}.png'))
                    utils.save_sketch(s_front, os.path.join(opt.outd, opt.outf, f'gt_s_front_{self.train_itr:06d}.png'))
                    self.summary_writer.add_scalar('train_p1/s_l1', s_l1.item(), self.train_itr)
                    self.summary_writer.add_scalar('train_p1/d_l1', d_l1.item(), self.train_itr)
                    self.summary_writer.add_scalar('train_p1/tv_l1', tv_l1.item(), self.train_itr)
                    # utils.plot_histogram(den_theta1, os.path.join(opt.outd, opt.outf, f'{self.train_itr:06d}_p1_h.png'))
                    
                den_prev = den_theta1
                gen_den_prev = gen_den_theta1
                s_front_prev = [s_front]

                # r_front_prev = r_front

                L = []
                L.append(w_d * d_l1 + w_sketch * s_l1 + w_tv * tv_l1)# + w_cls * cls_loss)

                for t in range(1, passes):
                    if self.use_sphere_mask:
                        den_theta2 = self.spatial_transformer(den_prev, augment_params['roll'+str(t+1)], augment_params['pitch'+str(t+1)], augment_params['yaw'+str(t+1)], order='rpy')
                    else:
                        den_theta2 = patch_sketcher.rotate(den_prev, augment_params['ortho'+str(t+1)])

                    s2 = self.render_sketches(den_theta2, rns=augment_params, views='zp')
                    if augment_params is not None and augment_params['noise'] == True:
                        s2_aug, = self._augment_sketches([s2], rns=augment_params)
                    else:
                        s2_aug = s2

                    if self.use_sphere_mask:
                        gen_den_theta2 = self.spatial_transformer(gen_den_prev, augment_params['roll'+str(t+1)], augment_params['pitch'+str(t+1)], augment_params['yaw'+str(t+1)], order='rpy')
                    else:
                        gen_den_theta2 = patch_sketcher.rotate(gen_den_prev, augment_params['ortho'+str(t+1)])

                    d_prev = gen_den_theta2

                    gen_den_theta2, residual_gen_den_theta2 = self.pix2vox_pair(gen_den_theta2, s2_aug)

                    gen_den_prev = gen_den_theta2

                    s2_recon = self.render_sketches(gen_den_theta2, rns=augment_params, views='zp')

                    # inv_s_l1 = patch_sketcher.inverse_sketch_loss(d=gen_den_theta2, dx=1, c=5, d_prev=d_prev, alpha=1)
                    # r2_recon = self.renderer(gen_den_theta2)
                    # r2 = self.renderer(den_theta2)
                    # proj_l1 = proj_l1 + losses.densityProjectionLoss(gen_den_theta2, den_theta2)
                    tv_l1_local = losses.totalVariation3D(gen_den_theta2)
                    # cls_loss_local = losses.clsLoss(gen_den_theta2, s2)

                    if self.use_sphere_mask:
                        gen_den_theta2 = self.spatial_transformer(gen_den_theta2, -augment_params['roll'+str(t+1)], -augment_params['pitch'+str(t+1)], -augment_params['yaw'+str(t+1)], order='ypr')
                    else:
                        gen_den_theta2 = patch_sketcher.rotate_back(gen_den_theta2, augment_params['ortho'+str(t+1)])

                    s2_front_recon = self.render_sketches(gen_den_theta2, rns=augment_params, views='zp')
                    # r2_front_recon = self.renderer(gen_den_theta2)

                    d_l1_local = (t + 1) * losses.densityLoss3D(gen_den_theta2, den_prev)

                    # grad_l1 = grad_l1 + losses.densityGradLoss(gen_den_theta2, den_prev)
                    # proj_l1 = proj_l1 + losses.densityProjectionLoss(gen_den_theta2, den_prev)
                    # print (len(s_front_prev))
                    # print (losses.sketchLoss(s2_recon, s2).item(), losses.sketchLoss(s2_front_recon, s_front_prev[-1]).item())
                    s_l1_local = losses.sketchLoss(s2_recon, s2) + w_prev * losses.sketchLoss(s2_front_recon, s_front_prev[-1])
                    tv_l1_local = losses.totalVariation3D(gen_den_theta2) + w_prev * losses.totalVariation3D(gen_den_prev)

                    # remember this ....
                    tmp_d = gen_den_theta2
                    for p in range(len(s_front_prev)-2,-1,-1):
                        if self.use_sphere_mask:
                            tmp_d = self.spatial_transformer(tmp_d, -augment_params['roll'+str(p+2)], -augment_params['pitch'+str(p+2)], -augment_params['yaw'+str(p+2)], order='ypr')
                        else:
                            tmp_d = patch_sketcher.rotate_back(tmp_d, augment_params['ortho'+str(p+2)])

                        sp = self.render_sketches(tmp_d, views='zp')
                        s_l1_local = s_l1_local + w_prev * losses.sketchLoss(sp, s_front_prev[p])
                        tv_l1_local = tv_l1_local + w_prev * losses.totalVariation3D(tmp_d)
                        # cls_loss_local = cls_loss_local + losses.clsLoss(tmp_d, s_front_prev[p])

                    # render_l1 = render_l1 + F.l1_loss(r2_recon, r2) + F.l1_loss(r2_front_recon, r_front_prev)
                    L.append(w_d * d_l1_local + w_sketch * s_l1_local + w_tv * tv_l1_local)# + w_cls * cls_loss_local)
                    
                    if self.train_itr % opt.log_freq == 0: # 100
                        # utils.save_render(self.renderer, gen_den_prev, os.path.join(opt.outd, opt.outf, f'{self.train_itr:06d}_p{t+1}_r_cur_recon.png'))
                        # utils.save_render(self.renderer, den_theta2, os.path.join(opt.outd, opt.outf, f'{self.train_itr:06d}_p{t+1}_r_cur_gt.png'))
                        # utils.save_render(self.renderer, gen_den_theta2, os.path.join(opt.outd, opt.outf, f'{self.train_itr:06d}_p{t+1}_r_prev_recon.png'))
                        # utils.save_render(self.renderer, den_prev, os.path.join(opt.outd, opt.outf, f'{self.train_itr:06d}_p{t+1}_r_prev_gt.png'))
                        # utils.save_sketch(s2_recon, os.path.join(opt.outd, opt.outf, f'{self.train_itr:06d}_p{t+1}_s_cur_recon.png'))
                        # utils.save_sketch(s2, os.path.join(opt.outd, opt.outf, f'{self.train_itr:06d}_p{t+1}_s_cur_gt.png'))
                        # utils.save_sketch(s2_front_recon, os.path.join(opt.outd, opt.outf, f'{self.train_itr:06d}_p{t+1}_s_prev_recon.png'))
                        # utils.save_sketch(s_front_prev[len(s_front_prev)-1], os.path.join(opt.outd, opt.outf, f'{self.train_itr:06d}_p{t+1}_s_prev_gt.png'))
                        # self.summary_writer.add_scalar('train_p'+str(t+1)+'/s_l1', s_l1.item()/(1+t*(opt.w_prev+1)), self.train_itr)
                        self.summary_writer.add_scalar('train_p'+str(t+1)+'/s_l1', s_l1_local.item(), self.train_itr)
                        self.summary_writer.add_scalar('train_p'+str(t+1)+'/d_l1', d_l1_local.item(), self.train_itr)
                        self.summary_writer.add_scalar('train_p'+str(t+1)+'/tv_l1', tv_l1_local.item(), self.train_itr)
                        self.summary_writer.add_scalar('w_prev', w_prev, self.train_itr)
                        # print (w_prev)

                    den_prev = den_theta2
                    s_front_prev.append(s2)
                    # r_front_prev = r2
                    
                # d_l1 = d_l1 / passes
                # # grad_l1 = grad_l1 / opt.passes
                # tv_l1 = tv_l1 / passes
                # s_l1 = s_l1 / passes
                # # proj_l1 = proj_l1 / (1+(passes-1)*2)

                ### aggregated losses
                d_loss = L[0]
                for t in range(1,len(L)):
                    # d_loss = d_loss + L[t] / (t + 1) # version 2
                    d_loss = d_loss + L[t] # version 1
                    

                d_loss = d_loss / ((1 + passes) * passes//2) # version 1


                # # render_l1 = render_l1 / (1+(opt.passes-1)*2)
                # d_loss = w_d * d_l1 + w_sketch * s_l1 + w_tv * tv_l1 #+ opt.w_inv_s * inv_s_l1 #+ w_grad * grad_l1
                # print (d_l1.item(), s_l1.item(), tv_l1.item())#, grad_l1.item(), tv_l1.item())
                # print (d_loss.item())
                d_loss.backward()
                self.optimizer_pix2vox_pair.step()
                
                if self.train_itr % opt.log_freq == 0:
                    # self.summary_writer.add_scalar('train_scale2/s_l1', s_l1.item(), self.train_itr)
                    # self.summary_writer.add_scalar('train_scale2/d_l1', d_l1.item(), self.train_itr)
                    # self.summary_writer.add_scalar('train_scale2/tv_l1', tv_l1.item(), self.train_itr)
                    # self.summary_writer.add_scalar('train_scale2/w_d', w_d, self.train_itr)
                    # self.summary_writer.add_scalar('train_scale2/w_sketch', w_sketch, self.train_itr)
                    # self.summary_writer.add_scalar('train_scale2/w_render', w_render, self.train_itr)
                    self.summary_writer.add_scalar('train_scale2/tooncolor', augment_params['tooncolor'], self.train_itr)
                    self.summary_writer.add_scalar('train_scale2/contour', augment_params['contour'], self.train_itr)
                    self.summary_writer.add_scalar('train_scale2/brightness', augment_params['brightness'], self.train_itr)
                    self.summary_writer.add_scalar('train_scale2/contrast', augment_params['contrast'], self.train_itr)
                    self.summary_writer.add_scalar('train_scale2/blur', augment_params['blur'], self.train_itr)
                    self.summary_writer.add_scalar('train_scale2/noise', augment_params['noise'], self.train_itr)

                    if opt.ngpus > 1:
                        torch.save(self.pix2vox_pair.module.state_dict(), os.path.join(opt.outd, opt.outm, f'pix2vox_pair_iter-{opt.max_epochs}.pth'))
                    else:
                        torch.save(self.pix2vox_pair.state_dict(), os.path.join(opt.outd, opt.outm, f'pix2vox_pair_iter-{opt.max_epochs}.pth')) 
                    
                    
                    self.test_synthetic(opt, opt.outd, self.train_itr)
                    # self.test(opt, opt.outd, self.train_itr)
                    # self.test_artist(opt, opt.outd, self.train_itr)
                    
                self.train_itr += 1
                # break
                
            self.summary_writer.add_scalar('misc/iter', self.train_itr, epoch)
            self.summary_writer.add_scalar('misc/epoch_time', (time()-t0)/60, epoch)
            epoch += 1

            