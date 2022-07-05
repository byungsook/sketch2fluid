
import matplotlib.pyplot as plt
import numpy as np
from perlin_numpy import generate_perlin_noise_3d

def perlin(oct, seed1, seed2, seed3):
    octaves = (oct, oct, oct)
    size = 129-(129%oct)+oct
    np.random.seed(seed1)
    noise1 = generate_perlin_noise_3d((size, size, size), octaves)[0:129,0:129,0:129]
    np.random.seed(seed2)
    noise2 = generate_perlin_noise_3d((size, size, size), octaves)[0:129,0:129,0:129]
    np.random.seed(seed3)
    noise3 = generate_perlin_noise_3d((size, size, size), octaves)[0:129,0:129,0:129]

    noise = np.array([noise1, noise2, noise3])
    return noise


def curlnoise(density, oct=10, seed1=1, seed2=10, seed3=100, vs=1, mask=True, show_vectorfield3d=False, show_vectorfield2d=False):
    round_x = np.copy(density)
    thres = round_x > 0
    anti = round_x <= 0
    round_x[anti] = 0
    round_x[thres] = 1
    if density is not None and not mask: noise =  np.array([density,density,density]) * perlin(oct, seed1, seed2, seed3)
    else: noise = perlin(oct, seed1, seed2, seed3)
    u, v, w = curl(noise)
    vel = np.array([u,v,w])*vs
    if mask:
        vel = np.array([round_x,round_x,round_x])*vel
    if show_vectorfield3d:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        x, y, z = np.meshgrid(np.arange(0, 129, 1),
                              np.arange(0, 129, 1),
                              np.arange(0, 129, 1))
        ax.quiver(x, y, z, vel[0], vel[1], vel[2], length=0.1)
        plt.show()

    if show_vectorfield2d:
        fig, ax = plt.subplots()
        utemp = vel[0, 64, :, :]
        vtemp = vel[1, 64, :, :]
        x, y = np.meshgrid(np.arange(0, 129, 1),
                           np.arange(0, 129, 1))

        ax.quiver(x,y,utemp,vtemp)
        plt.show()

    return vel

def curl(x):
    dwdy = x[2, :, 1:, :] - x[2, :, :-1, :]
    dvdz = x[1, :, :, 1:] - x[1, :, :, :-1]
    dudz = x[0, :, :, 1:] - x[0, :, :, :-1]
    dwdx = x[2, 1:, :, :] - x[2, :-1, :, :]
    dvdx = x[1, 1:, :, :] - x[1, :-1, :, :]
    dudy = x[0, :, 1:, :] - x[0, :, :-1, :]

    dwdy = np.concatenate((dwdy, np.expand_dims(dwdy[:,-1,:], axis=1)), axis=1)
    dvdz = np.concatenate((dvdz, np.expand_dims(dvdz[:,:,-1], axis=2)), axis=2)
    dudz = np.concatenate((dudz, np.expand_dims(dudz[:,:,-1], axis=2)), axis=2)
    dwdx = np.concatenate((dwdx, np.expand_dims(dwdx[-1,:,:], axis=0)), axis=0)
    dvdx = np.concatenate((dvdx, np.expand_dims(dvdx[-1,:,:], axis=0)), axis=0)
    dudy = np.concatenate((dudy, np.expand_dims(dudy[:,-1,:], axis=1)), axis=1)

    u = dwdy - dvdz
    v = dudz - dwdx
    w = dvdx - dudy
    return u,v,w