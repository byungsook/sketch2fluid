import torch
import torch.nn as nn

if torch.cuda.is_available():
  dev = "cuda:0"
else:
  dev = "cpu"
device = torch.device(dev)


class ResBlock(nn.Module):
    """ Residual block used in the generator """
    def __init__(self, inChannels, midChannels, outChannels, BN=False, residual=False):
        super().__init__()
        self.inChannels = inChannels
        self.midChannels = midChannels
        self.outChannels = outChannels
        self.BN = BN
        self.residual = residual


        self.conv_block = self.build_conv_block()
        if self.BN:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels=self.inChannels, out_channels=self.outChannels, kernel_size=1, stride=1),
                                          nn.BatchNorm2d(self.outChannels))
        else:
            self.shortcut = nn.Conv2d(in_channels=self.inChannels, out_channels=self.outChannels, kernel_size=1, stride=1)

    def build_conv_block(self):
        conv_block = []
        conv_block += [nn.ReflectionPad2d(2)]
        conv_block += [nn.Conv2d(in_channels=self.inChannels, out_channels=self.midChannels, kernel_size=5, stride=1, padding=0)]  # TODO: check parameters
        if self.BN: conv_block += [nn.BatchNorm2d(self.midChannels)]
        if self.residual: conv_block += [nn.Tanh(), nn.Dropout(0.5)]
        else: conv_block += [nn.ReLU(True), nn.Dropout(0.5)]
        conv_block += [nn.ReflectionPad2d(2)]
        conv_block += [nn.Conv2d(in_channels=self.midChannels, out_channels=self.outChannels, kernel_size=5, stride=1, padding=0)]  # TODO: check parameters
        if self.BN: conv_block += [nn.BatchNorm2d(self.outChannels)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = self.shortcut(x).to(device) + self.conv_block(x).to(device)
        return nn.functional.relu(out).to(device)


class Generator(nn.Module):

    def __init__(self, channels, superRes=True, BN=True):
        super().__init__()
        self.superRes = superRes
        self.BN = BN

        model = []
        model += [ResBlock(inChannels=channels, midChannels=8, outChannels=32, BN=self.BN)]
        model += [ResBlock(inChannels=32, midChannels=128, outChannels=128, BN=self.BN)]
        model += [ResBlock(128, 32, 8, BN=self.BN)]
        model += [ResBlock(8, 2, 1, BN=False)]
        #model += [nn.AvgPool2d(2, 2)]
        self.netG = nn.Sequential(*model).to(device)


    def forward(self, x):
        if self.superRes: out = nn.functional.interpolate(x, scale_factor=4)
        else: out = x #nn.functional.interpolate(x, scale_factor=2)
        out = self.netG(out).to(device)
        return out

class Discriminator(nn.Module):

    def __init__(self, channels=2, BN=False, superRes=True):
        super().__init__()
        self.BN = BN
        self.channels = channels
        self.superRes = superRes
        """
        model1 = [nn.ReflectionPad2d(1)]
        model1 += [nn.Conv2d(in_channels=2, out_channels=32, kernel_size=4, stride=2, padding=0)]  # TODO: check parameters
        model1 += [nn.LeakyReLU(True)]
        self.d1 = nn.Sequential(*model1).to(device)

        model2 = [nn.ReflectionPad2d(1)]
        model2 += [nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0)]
        if self.BN: model2 += [nn.BatchNorm2d(64)]
        model2 += [nn.LeakyReLU(True)]
        self.d2 = nn.Sequential(*model2).to(device)

        model3 = [nn.ReflectionPad2d(1)]
        model3 += [nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=0)]
        if self.BN: model3 += [nn.BatchNorm2d(128)]
        model3 += [nn.LeakyReLU(True)]
        self.d3 = nn.Sequential(*model3).to(device)

        model4 = [nn.ReflectionPad2d((1, 2, 1, 2))]
        model4 += [nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=1)]
        if self.BN: model4 += [nn.BatchNorm2d(256)]
        model4 += [nn.LeakyReLU(True)]
        self.d4 = nn.Sequential(*model4).to(device)
        """
        model = [nn.ReflectionPad2d(1)]
        model += [nn.Conv2d(in_channels=2, out_channels=32, kernel_size=4, stride=2, padding=0)]  # TODO: check parameters
        model += [nn.LeakyReLU(True)]

        model += [nn.ReflectionPad2d(1)]
        model += [nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0)]
        if self.BN: model += [nn.BatchNorm2d(64)]
        model += [nn.LeakyReLU(True)]

        model += [nn.ReflectionPad2d(1)]
        model += [nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=0)]
        if self.BN: model += [nn.BatchNorm2d(128)]
        model += [nn.LeakyReLU(True)]

        model += [nn.ReflectionPad2d((1, 2, 1, 2))]
        model += [nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=1)]
        if self.BN: model += [nn.BatchNorm2d(256)]
        model += [nn.LeakyReLU(True)]

        model += [nn.Flatten()]
        model += [nn.Linear(16384,1)]
        model += [nn.Sigmoid()]
        self.d = nn.Sequential(*model).to(device)



    def forward(self, x, y):
        if self.superRes:
            lR = nn.functional.interpolate(x, scale_factor=4)
        else:
            lR = x
        input = torch.cat((lR, y), dim=1)
        """
        d1 = self.d1(input).to(device)
        d2 = self.d2(d1).to(device)
        d3 = self.d3(d2).to(device)
        d4 = self.d4(d3).to(device)
        out = d4.view(d4.shape[0], -1)
        """
        out = self.d(input)

        # out = self.netD(input)
        #linear = nn.Linear(out.shape[1], 1).to(device)
        #sigmoid = nn.Sigmoid()

        return out, 0, 0, 0, 0
