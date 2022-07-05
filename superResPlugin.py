import torch
import gc
import numpy as np
import cupy as cp
import cupyx.scipy.ndimage

from superRes.models import Generator
from superRes.curlnoise import curlnoise

if torch.cuda.is_available():
  dev = "cuda:0"
else:
  dev = "cpu"
device = torch.device(dev)
print(device)


def SuperRes(mapfile, res, outputPath, seed, oct, vel_scale):
    seed = int(seed)
    oct = int(oct)
    vel_scale = int(vel_scale)
    fp = np.memmap(mapfile, dtype='float32', mode='readwrite', shape=res)

    generator1 = Generator(channels=4, superRes=False)
    generator2 = Generator(channels=4, superRes=False)
    print('===> Load checkpoints ...')
    if device == 'cpu':
        generator1.load_state_dict(torch.load('./superRes/ckt/test0244/gen_epoch_20.pth'), map_location={'cuda:0': 'cpu'})
        generator2.load_state_dict(torch.load('./superRes/ckt/test0229/gen_epoch_20.pth'), map_location={'cuda:0': 'cpu'})
    else:
        generator1.load_state_dict(torch.load('./superRes/ckt/test0244/gen_epoch_20.pth'))
        generator2.load_state_dict(torch.load('./superRes/ckt/test0229/gen_epoch_20.pth'))

    x = np.reshape(fp, (1,129,129,129,1))


    velocity = curlnoise(x[0,:,:,:,0], seed1=1+seed, seed2=10+seed, seed3=100+seed, oct=oct , vs=vel_scale)
    velocity = np.reshape(velocity, (3, 129, 129, 129, 1))
    velocity = np.swapaxes(velocity, 0, 4).astype(np.float32)
    x = np.concatenate((x, velocity), axis=4)

    output, shape = upsample(x_3d=x, gen=generator1, size=129,
                             channels=4, threshold=0.001)

    print("second pass")
    output = np.concatenate([output.reshape(1, shape[0], shape[1], shape[2], 1), velocity], axis=4)
    output, shape = upsample(x_3d=output, gen=generator2, size=129,
                             channels=4, threshold=0.001, mpass=2)

    fp[:] = output[:]

    return mapfile, shape


def upsample(x_3d, gen, batchSize=10, size=129, channels=4, threshold=0.001, mpass=1):

    intermRes = []

    up = 1
    batch_xs_tile = x_3d
    if mpass == 2: batch_xs_tile = np.swapaxes(batch_xs_tile, 1, 3)
    batch_xs_in = np.swapaxes(batch_xs_tile, 0, 1)
    batch_xs_in = np.swapaxes(batch_xs_in, 1, 4)

    for j in range(0, batch_xs_in.shape[0] // batchSize):
        # print("percentage: "+ str(100*j/(batch_xs_in.shape[0]//batchSize))+"%")
        inputData = batch_xs_in[j * batchSize:(j + 1) * batchSize].reshape(
            [batchSize, channels, size, size])

        if (np.average(inputData[:, 0, :, :]) >= threshold):
            inputData = torch.from_numpy(inputData).to(device)
            result = gen(inputData)
            intermRes.extend(result.detach().cpu().numpy())
        else:
            temp = cp.asarray(inputData[:, 0:1, :, :])
            result = cp.asnumpy(cupyx.scipy.ndimage.zoom(temp, (1, 1, up, up)))
            intermRes.extend(result)

        inputData = []
        result = []
        gc.collect()

    if len(intermRes) < len(batch_xs_in):
        inputData = batch_xs_in[len(intermRes):len(batch_xs_in)].reshape(
            [len(batch_xs_in) - len(intermRes), channels, size, size])

        if (np.average(inputData) >= threshold):
            inputData = torch.from_numpy(inputData).to(device)
            result = gen(inputData)
            # print(result.shape)
            intermRes.extend(result.detach().cpu().numpy())
        else:
            temp = cp.asarray(inputData[:, 0:1, :, :])
            result = cp.asnumpy(cupyx.scipy.ndimage.zoom(temp, (1, 1, up, up)))
            intermRes.extend(result)

        inputData = []
        result = []
        gc.collect()

    shape = (size * up, size * up, size * up)
    output = np.reshape(intermRes, shape)
    if mpass == 2: output = np.swapaxes(output, 0, 2)

    return output, shape