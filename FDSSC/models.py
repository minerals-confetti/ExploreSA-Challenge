import torch
import torch.nn as nn
import torch.nn.functional as F


class Bn_prelu(nn.Module):

    def __init__(self):
        super(Bn_prelu, self).__init__()
        self.bn = nn.BatchNorm3d(num_features=1)
        self.prelu = nn.PReLU()
    
    def forward(self, x):
        x = self.bn(x)
        x = self.prelu(x)
        return x

class Spectral_conv(nn.Module):
    #NEED TO IMPLEMENT PADDING
    def __init__(self, input_channels):
        super(Spectral_conv, self).__init__()
        self.bn_prelu = Bn_prelu()
        self.conv3d = nn.Conv3d(input_channels, growth_rate, kernel_size=(1, 1, 7))

    def forward(self, x):
        x = self.bn_prelu(x)
        x = self.conv3d(x)
        return x

class Spatial_conv(nn.Module):

    def __init__(self, input_channels):
        super(Spatial_conv, self).__init__()
        self.bn_prelu = Bn_prelu()
        self.conv3d = nn.Conv3d(input_channels, growth_rate, kernel_size=(3, 3, 1))

    def forward(self, x):
        x = self.bn_prelu(x)
        x = self.conv3d(x)
        return x

class FDSSC_model(nn.Module):

    def __init__(self, input_shape):
        '''Input shape in the form of (N,C,H,W,D)'''
        super(FDSSC_model, self).__init__()
        self.input_shape = input_shape

        global growth_rate
        growth_rate = 12

        self.conv3D1 = nn.Conv3d(1, 24, kernel_size=(1, 1, 7), stride=(1, 1, 2))
        self.spectral_conv1 = Spectral_conv(24)
        self.spectral_conv2 = Spectral_conv(24)


    
    