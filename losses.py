
import torch
import torch.nn.functional as F

# from utils import *
import patch_sketcher

def kldiv(mu, logvar):
    kld = -0.5 * torch.mean(torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    return kld


def densityGradLoss(x_recon, x):
    '''
        input: x [0,1]
    '''
    recon_j = patch_sketcher.torchGrad3D(x_recon, order=2)
    x_j = patch_sketcher.torchGrad3D(x, order=2)
    loss = F.l1_loss(recon_j, x_j)
    return loss

# def gradient_loss_3d_vel(x_recon, x):
#     recon_j, _ = torchJacobian3(x_recon)
#     x_j, _ = torchJacobian3(x)
#     loss = F.l1_loss(recon_j, x_j)
#     return loss



def densityLoss3D(x_recon, x):
    '''
    Compute 3D density loss
    input: tensor of shape (B, C, D, H, W).
    '''
    return F.l1_loss(x_recon, x)

def totalVariation2D(x_recon, beta=4):
    '''
    Compute 2D total variation loss.
    input: tensor of shape (B, C, H, W).
    '''
    w_variance = torch.sum(torch.pow(x_recon[:, :, :, :-1] - x_recon[:, :, :, 1:], beta/2))
    h_variance = torch.sum(torch.pow(x_recon[:, :, :-1, :] - x_recon[:, :, 1:, :], beta/2))
    return (w_variance + h_variance) / x_recon.size(0)


def totalVariation3D(x_recon, mode=0):
    '''
    Compute 2D total variation loss.
    input: tensor of shape (B, C, D, H, W).
    mode:
        0 for depth only

    '''
    # w_variance = torch.mean(torch.pow(x_recon[:, :, :, :, :-1] - x_recon[:, :, :, :, 1:], beta/2))
    # h_variance = torch.mean(torch.pow(x_recon[:, :, :, :-1, :] - x_recon[:, :, :, 1:, :], beta/2))
    # d_variance = torch.mean(torch.pow(x_recon[:, :, :-1, :, :] - x_recon[:, :, 1:, :, :], beta/2))
    
    w_variance = torch.mean(torch.abs(x_recon[:, :, :, :, :-1] - x_recon[:, :, :, :, 1:]))
    h_variance = torch.mean(torch.abs(x_recon[:, :, :, :-1, :] - x_recon[:, :, :, 1:, :]))
    d_variance = torch.mean(torch.abs(x_recon[:, :, :-1, :, :] - x_recon[:, :, 1:, :, :]))
    
    if mode == 0:
        l = d_variance
    else:
        l = w_variance + h_variance + d_variance
    return l




def densityProjectionLoss(x_recon, x):
    '''
    Compute 3D density projection loss
    input: tensor of shape (B, C, D, H, W).
    '''
    return F.l1_loss(torch.mean(x_recon,2), torch.mean(x,2))


def sketchLoss(x_recon, x):
    '''
    Compute 2D sketch l1 loss
    input: tensor of shape (B, C, H, W).
    '''
    return F.l1_loss(x_recon, x)


def clsLoss(d, s):
    '''
    Classification loss
    input: density and sketch
    '''
    s = F.interpolate(s, size=d.size(-1), mode='bilinear')
    s[s<1]=0
    mask = s * torch.sum(d,2)
    # print (mask.size())
    # import matplotlib.pyplot as plt
    # plt.figure(1)
    # plt.imshow(mask.detach().cpu().numpy()[0,0], cmap=plt.cm.gnuplot2)
    # plt.show()

    

    return torch.mean(mask)




