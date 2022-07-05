from collections import OrderedDict
import torch
import torch.nn as nn
#from torchvision import models


def conv2d(nin, nout, activ, ks=3, s=2, p=1, bn=False, pn=False):
    conv = nn.Conv2d(nin, nout, kernel_size=ks, stride=s, padding=p)
    layers = OrderedDict()
    layers['conv'] = conv

    if bn:
        layers['bn'] = nn.BatchNorm2d(nout)
    if activ:
        layers['activ'] = activ

    if pn:
        layers['pn'] = PixelNormLayer()

    return nn.Sequential(layers)

def conv3d(nin, nout, activ, ks=3, s=2, p=1, bn=False, pn=False):
    conv = nn.Conv3d(nin, nout, kernel_size=ks, stride=s, padding=p)
    layers = OrderedDict()
    layers['conv'] = conv

    if bn:
        layers['bn'] = nn.BatchNorm3d(nout)
    if activ:
        layers['activ'] = activ

    if pn:
        layers['pn'] = PixelNormLayer()

    return nn.Sequential(layers)


def up_conv3d(nin, nout, activ, ks=3, s=1, p=1, up_factor=2, bn=False, pn=False):
    layers = OrderedDict()

    up = nn.Upsample(scale_factor=up_factor, mode='nearest')
    layers['up'] = up
    
    conv = nn.Conv3d(nin, nout, kernel_size=ks, stride=s, padding=p)
    layers['conv'] = conv

    if bn:
        layers['bn'] = nn.BatchNorm3d(nout)
    if activ:
        layers['activ'] = activ

    if pn:
        layers['pn'] = PixelNormLayer()
    return nn.Sequential(layers)

def pad_conv3d(nin, nout, activ, ks=3, s=1, p=1, bn=False, pn=False):
    layers = OrderedDict()

    pd = nn.ReplicationPad3d(1)
    layers['pd'] = pd

    conv = nn.Conv3d(nin, nout, kernel_size=ks, stride=s, padding=p)
    layers['conv'] = conv

    if bn:
        layers['bn'] = nn.BatchNorm3d(nout)
    if activ:
        layers['activ'] = activ

    if pn:
        layers['pn'] = PixelNormLayer()
    return nn.Sequential(layers)

def up_pad_conv3d(nin, nout, activ, ks=3, s=1, p=1, up_factor=2, bn=False, pn=False):
    layers = OrderedDict()

    up = nn.Upsample(scale_factor=up_factor, mode='nearest')
    layers['up'] = up

    pd = nn.ReplicationPad3d(1)
    layers['pd'] = pd

    conv = nn.Conv3d(nin, nout, kernel_size=ks, stride=s, padding=p)
    layers['conv'] = conv

    if bn:
        layers['bn'] = nn.BatchNorm3d(nout)
    if activ:
        layers['activ'] = activ

    if pn:
        layers['pn'] = PixelNormLayer()
    return nn.Sequential(layers)


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

def conv(nin, nout, kernel_size=3, stride=1, padding=1, layer=nn.Conv3d,
         ws=False, bn=False, pn=False, activ=None, gainWS=2):
    conv = layer(nin, nout, kernel_size, stride=stride, padding=padding, bias=False if bn else True)
    layers = OrderedDict()

    if ws:
        layers['ws'] = WScaleLayer(conv, gain=gainWS)

    layers['conv'] = conv

    if bn:
        layers['bn'] = nn.BatchNorm3d(nout)
    if activ:
        if activ == nn.PReLU:
            # to avoid sharing the same parameter, activ must be set to nn.PReLU (without '()') and initialized here
            layers['activ'] = activ(num_parameters=1)
        else:
            layers['activ'] = activ
    if pn:
        layers['pn'] = PixelNormLayer()
    return nn.Sequential(layers)


def linear(nin, nout, bn=False, activ=None):
    fc = nn.Linear(nin, nout)
    layers = OrderedDict()
    layers['fc'] = fc
    if bn:
        layers['bn'] = nn.BatchNorm1d(nout)
    if activ:
        layers['activ'] = activ
    
    return nn.Sequential(layers)


def conv_transpose(nin, nout, kernel_size=4, stride=2, padding=1, layer=nn.ConvTranspose3d,
         ws=False, bn=False, pn=False, activ=None, gainWS=2):
    conv = layer(nin, nout, kernel_size, stride=stride, padding=padding, bias=False if bn else True)
    layers = OrderedDict()

    layers['conv'] = conv

    if bn:
        layers['bn'] = nn.BatchNorm3d(nout)
    if activ:
        if activ == nn.PReLU:
            # to avoid sharing the same parameter, activ must be set to nn.PReLU (without '()') and initialized here
            layers['activ'] = activ(num_parameters=1)
        else:
            layers['activ'] = activ
    
    return nn.Sequential(layers)

def conv2d_transpose(nin, nout, activ=None, ks=4, s=2, p=1, bn=False, dropout=None):

    layers = OrderedDict()

    conv = nn.ConvTranspose2d(nin, nout, ks, stride=s, padding=p)
    layers['conv'] = conv

    if bn:
        layers['bn'] = nn.BatchNorm2d(nout)
    if dropout is not None and dropout > 0:
        layers['dropout'] = nn.Dropout(dropout)
    if activ:
        layers['activ'] = activ
    
    return nn.Sequential(layers)

def conv2d_up(nin, nout, activ=None, ks=3, s=1, p=1, bn=False, scale=2, dropout=None):

    layers = OrderedDict()

    up = nn.Upsample(scale_factor=scale, mode='nearest')
    layers['up'] = up
    
    conv = nn.Conv2d(nin, nout, ks, stride=s, padding=p)
    layers['conv'] = conv

    if bn:
        layers['bn'] = nn.BatchNorm2d(nout)
    if dropout is not None and dropout > 0:
        layers['dropout'] = nn.Dropout(dropout)
    if activ:
        layers['activ'] = activ
    
    return nn.Sequential(layers)

def conv3d_transpose(nin, nout, activ=None, ks=4, s=2, p=1, bn=False, dropout=None):

    layers = OrderedDict()

    conv = nn.ConvTranspose3d(nin, nout, ks, stride=s, padding=p)
    layers['conv'] = conv

    if bn:
        layers['bn'] = nn.BatchNorm3d(nout)
    if dropout is not None and dropout > 0:
        layers['dropout'] = nn.Dropout(dropout)
    if activ:
        layers['activ'] = activ
    
    return nn.Sequential(layers)

class PixelNormLayer(nn.Module):
    def __init__(self):
        super(PixelNormLayer, self).__init__()

    def forward(self, x):
        # seems correct
        pn = x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)
        # print (pn.mean())
        # pn = x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)
        # print (pn.mean())
        return pn

    def __repr__(self):
        return self.__class__.__name__


class WScaleLayer(nn.Module):
    def __init__(self, incoming, gain=2):
        super(WScaleLayer, self).__init__()

        self.gain = gain
        self.scale = (self.gain / incoming.weight[0].numel()) ** 0.5    # seems work
        # print (self.scale)
        # self.scale = 1

        # self.scale = (torch.mean(incoming.weight.data ** 2)) ** 0.5
        # # self.incoming.weight.data.copy_(self.incoming.weight.data / self.scale)
        # print (self.scale)
    def forward(self, input):
        return input * self.scale

    def __repr__(self):
        return '{}(gain={})'.format(self.__class__.__name__, self.gain)



def load_pretrained_models(device, enc_type='densenet', is_grad=False, verbose=False):

    if enc_type == 'resnet':
        pretrained_model = models.resnet18(pretrained=True)   # have some results
        # pretrained_model = models.resnet34(pretrained=True) # new
        # print (pretrained_model)
        fe = nn.Sequential(
            pretrained_model.conv1,
            pretrained_model.bn1,
            pretrained_model.relu,
            pretrained_model.maxpool,
            *list(pretrained_model.layer1.children())[:],
            *list(pretrained_model.layer2.children())[:],
            *list(pretrained_model.layer3.children())[:],
        )
        # print (pretrained_model)
        
    elif enc_type == 'densenet':
        # pretrained_model = models.densenet121(pretrained='imagenet')
        # pretrained_model = models.densenet121(pretrained=True)
        
        # print (pretrained_model)
        # fe = nn.Sequential(
        #     pretrained_model.features.conv0,
        #     pretrained_model.features.norm0,
        #     pretrained_model.features.relu0,
        #     pretrained_model.features.pool0,
        #     *list(pretrained_model.features.denseblock1.children())[:],
        #     *list(pretrained_model.features.transition1.children())[:],
        #     *list(pretrained_model.features.denseblock2.children())[:],
        #     *list(pretrained_model.features.transition2.children())[:],
        # )

        pretrained_model = models.densenet161(pretrained=True)
        # print (pretrained_model)
        fe = nn.Sequential(
            pretrained_model.features.conv0,
            pretrained_model.features.norm0,
            pretrained_model.features.relu0,
            pretrained_model.features.pool0,
            *list(pretrained_model.features.denseblock1.children())[:],
            *list(pretrained_model.features.transition1.children())[:],
            *list(pretrained_model.features.denseblock2.children())[:],
            *list(pretrained_model.features.transition2.children())[:],
            *list(pretrained_model.features.denseblock3.children())[:],
            *list(pretrained_model.features.transition3.children())[:],
            *list(pretrained_model.features.denseblock4.children())[:],
            *list(pretrained_model.features.transition4.children())[:],
               
        )

    
    elif enc_type == 'vgg':
        layer = 15 # 19
        pretrained_model = models.vgg16(pretrained=True)
        # print (pretrained_model)
        fe = nn.Sequential(*list(pretrained_model.features.children())[:layer+1])

    else:
        raise ValueError('encoder type should be among fe, vgg or resnet')
    
    if verbose:
        print (pretrained_model)
    
    for param in fe.parameters():
        param.requires_grad = is_grad

    fe = fe.to(device)
    fe.eval()
    return fe



