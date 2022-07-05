import argparse
import numpy as np

import os
import re
import math

import scipy.ndimage as snd
from matplotlib import pyplot as plt

from multiprocessing import Pool

import torch
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

import time

class Timer:    
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = self.end - self.start

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--data', default='../local_out/DDV_test/dissolve', help='directory of the keyframes')
    parser.add_argument('--out', default='../local_out/DDV_test/out/dissolve', help='base directory where the result is stored')
    parser.add_argument('--frames', type = int, default=2, help="number of intermediate frames, including one of the keyframes")
    parser.add_argument('--processes', type = int, default=20, help="number of preocesses")
    parser.add_argument('--sigma', type=float, default=10, help='sigma')
    parser.add_argument('--niter', type = int, default=500, help="number of iterations")
    parser.add_argument('--kframe_start', type = int, default=0, help="skip frames at load time")
    parser.add_argument('--kframe_stop', type = int, default=None, help="skip frames at load time")
    parser.add_argument('--kframe_step', type = int, default=1, help="skip frames at load time")
    parser.add_argument('--save_each', type = int, default=None, help="save each n iterations")
    parser.add_argument('--linear', action='store_true', help='linear interpolation')

    # use linear for now.
    # cubic implem is here: https://hal.archives-ouvertes.fr/hal-01682107v2/document 
    # for later ref: https://hal.archives-ouvertes.fr/hal-01621311v2/document

    opt.linear = True

    opt = parser.parse_args()

    files = [p for p in (os.path.join(opt.data, f) for f in os.listdir(opt.data)) if os.path.isfile(p) ]

    files = [f for f in files if 'fl_' in f]

    files.sort(key=lambda f: int(re.sub('\D', '', f)))
    files = files[opt.kframe_start:opt.kframe_stop:opt.kframe_step]

    frame_list = np.arange(opt.frames * (len(files)-1)+1)

    if opt.linear:
        files_list = [
            [files[kf], files[kf+1]]
            for kf in [
                min(i//opt.frames, len(files)-2) for i in frame_list
            ]
        ]
    else:

        files.insert(0, files[0])
        files.append(files[-1])

        files_list = [
            [files[kf], files[kf+1], files[kf+2], files[kf+3]]
            for kf in [
                min(i//opt.frames, len(files)-4) for i in frame_list
            ]
        ]

    x_list = (frame_list % opt.frames) / opt.frames
    x_list[-1] = 1

    if not os.path.exists(opt.out):
        os.mkdir(opt.out)

    save_list = [
        os.path.join(opt.out, f'{f}.npz') for f in frame_list
    ]
    
    logger = [f"Frame {f}:" for f in frame_list]
    
    # with Pool(processes=opt.processes) as pool:
    #     pool.starmap(
    #         run,
    #         zip(
    #         *(files_list,
    #         save_list,
    #         x_list,
    #         (opt.sigma,)*len(x_list),
    #         (opt.niter,)*len(x_list),
    #         logger,
    #         (opt.save_each,)*len(x_list)
    #         ))
    #     )

    print(files_list)

    map(run,
            zip(
            *(files_list,
            save_list,
            x_list,
            (opt.sigma,)*len(x_list),
            (opt.niter,)*len(x_list),
            logger,
            (opt.save_each,)*len(x_list)
            ))
    )

    outr = os.path.join(opt.out, 'render') 

    #print((f'start "" python dataset/render_all.py --data {outd} --out {outr} --video'))
    # os.system(f'bsub -W 3:59 -G ls_grossm -R "rusage[mem=4096]" python dataset/render_all.py --data {opt.out} --out {outr} --video --grid --density_mult 2')

    #bsub -W 3:59 -G ls_grossm -R "rusage[mem=4096]" python dataset/render_all.py --data link_share/results/OT/6dissolve --out link_share/results/OT/6dissolve/render --video --grid --density_mult 2
    # bsub -W 3:59 -G ls_grossm -R "rusage[mem=4096]" python dataset/render_all.py --data link_share/results/OT/6turbulent --out link_share/results/OT/6turbulent/render --video --grid --density_mult 1.5



def run(file_list, out_file, x, sigma, niter, log_info, save_each):
    print("start")

    dlist = list([
        data[ 'd' if 'd' in data.keys() else "density"]
        for data in [
            np.load(file) for file in file_list
        ] 
    ])

    with Timer() as t:

        if len(dlist) == 2:
            #linear interpolation
            weights = np.array([1-x, x])

        else:
            #cubic interpolation

            # m0 = p1 - p-1 ; m1 = p2 - p0
            # y = h00 p0 + h10 m0 + h01 p1 + h11 m1
            # y = p-1 (-h10) + p0 (h00 - h11) + p1 (h01 + h10) + p2(h11)

            h00 = 2*x**3 - 3*x**2 + 1
            h10 = x**3 - 2*x**2 + x
            h01 = -2*x**3 + 3*x**2
            h11 = x**3 - x**2

            weights = np.array([-h10, h00 - h11, h01 + h10, h11])

        sum_list = [np.sum(d) for d in dlist]
        dlist = np.array([d/np.sum(d) for d in dlist])

        target_scale =  np.dot(weights, sum_list)

        b = w_barry_log_stable(dlist, weights, sigma, niter=niter, tol=0, save_each = save_each, target_scale = target_scale, filename = out_file)

    print(log_info, f"Computed barycenter {x} : recovery {np.sum(b)}, time {t.interval}")

    #np.savez_compressed(out_file, density = b)

def gaussian_kernel(a, eps):
    shape = a.shape[0]
    sigma = math.sqrt(eps/2)
    truncate = shape / sigma
    normalization =  np.sum(np.exp(-np.arange(-shape,shape)**2/eps))**len(a.shape)
    return lambda a: snd.gaussian_filter(a, sigma, truncate = truncate)*normalization

def torch_gaussian(x, eps):

    # In, for now: C, D, H, W

    transpose = (0,) + tuple(1+(np.arange(1, len(x.shape)+1) % len(x.shape)))
    d_shape = (None, slice(None), *((None, )*(len(x.shape)-1)))

    #21 22, 21

    xx = torch.arange(x.shape[0]).to(DEVICE)

    for _ in range(len(x.shape)):
        x = torch.permute(x, axes = transpose)
        out = x.copy()

        for i in range(x.shape[0]):

            s = slice(min(0, i-15), max(x.shape[0], i+15))

            d = (i - xx[s])**2
            out[:, i] = torch.sum(x[:, s] * torch.exp(-d[d_shape]/eps), dim = 0)

        x = out

    return x


def safe_log(x):
    return torch.log(torch.max(x, torch.finfo(x.dtype).tiny))


def w_barry_log_stable(mu_list, weights, sigma, niter=100, tol=1e-5, beta=None, out = None, save_each = None, target_scale = 1, filename = ""):

    mu_list += 1e-30

    if beta is None:
        beta = torch.zeros_like(mu_list)
        
    def MKD(x):
        return sigma * safe_log(torch_gaussian(torch.exp(x/sigma)))

    log_mu = sigma*torch.log(mu_list)

    weights = weights[(slice(None), *((None,)*len(mu_list[0].shape)))]
        
    for i in range(niter):

        alpha = log_mu - MKD(beta)

        conv =  MKD(alpha)
        vconv = beta + conv
        
        mu = np.sum(weights * vconv, axis = 0)

        beta = mu[None] - MKD(alpha)

        if (save_each is not None and (i+1)%save_each == 0) or i == niter-1:
            mu_rec = np.exp(mu/sigma)
            np.savez_compressed(filename,
                density = mu_rec / np.sum(mu_rec) * target_scale,
                beta = beta,
                niter = niter
            )
        
    # print(sigma, i, np.sqrt(err))
    # return mu_rec
    return torch.exp(mu/sigma)


if __name__ == "__main__":
    main()