import torch
import torch.nn as nn
import torch.nn.functional as F
from . import utils

class Bn_prelu(nn.Module):

    def __init__(self, num_features):
        super(Bn_prelu, self).__init__()
        self.bn = nn.BatchNorm3d(num_features=num_features)
        self.prelu = nn.PReLU()
    
    def forward(self, x):
        x = self.bn(x)
        x = self.prelu(x)
        return x

class Spectral_conv(nn.Module):
    #NEED TO IMPLEMENT PADDING
    def __init__(self, input_channels):
        super(Spectral_conv, self).__init__()
        self.bn_prelu = Bn_prelu(input_channels)
        self.conv3d = nn.Conv3d(input_channels, growth_rate, kernel_size=(1, 1, 7), padding=(0, 0, 3))

    def forward(self, x):
        x = self.bn_prelu(x)
        x = self.conv3d(x)
        return x

class Spatial_conv(nn.Module):

    def __init__(self, input_channels):
        super(Spatial_conv, self).__init__()
        self.bn_prelu = Bn_prelu(input_channels)
        self.conv3d = nn.Conv3d(input_channels, growth_rate, kernel_size=(3, 3, 1), padding=(1, 1, 0))

    def forward(self, x):
        x = self.bn_prelu(x)
        x = self.conv3d(x)
        return x

class FDSSC_model(nn.Module):

    def __init__(self, input_shape, classes):
        '''Input shape in the form of (C,H,W,D)'''
        super(FDSSC_model, self).__init__()
        self.input_shape = input_shape

        global growth_rate
        growth_rate = 12

        # add padding since original paper would skip over channels that don't divide evenly into the convolution
        inp_padding = utils.ceildiv((5 - input_shape[3] % 2), 2) # 5 = 7 - 2 finding padding with formula 2P = 2 * out - inp + 7(filter size) - 2(strides)
        depth_after_spec = utils.ceildiv(input_shape[3], 2)
        # definitions for dense spectral layers
        self.input_spec_conv = nn.Conv3d(1, 24, kernel_size=(1, 1, 7), stride=(1, 1, 2), padding=(0, 0, inp_padding))
        self.spectral_conv1 = Spectral_conv(24)
        self.spectral_conv2 = Spectral_conv(36)
        self.spectral_conv3 = Spectral_conv(48)
        self.bn_prelu1 = Bn_prelu(60)

        #reshaping and transforming to prep for spatial conv
        conv_depth = depth_after_spec # maybe consider setting this to an integer so we can apply transfer learning? (need to pad as well)
        self.reshape_conv = nn.Conv3d(60, 200, kernel_size=(1, 1, conv_depth), stride=(1, 1, 1))
        self.bn_prelu2 = Bn_prelu(200)

        # definitions for dense spatial layers
        self.input_spat_conv = nn.Conv3d(1, 24, kernel_size=(3, 3, 200), stride=(1, 1, 1))
        self.spatial_conv1 = Spatial_conv(24)
        self.spatial_conv2 = Spatial_conv(36)
        self.spatial_conv3 = Spatial_conv(48)
        self.bn_prelu3 = Bn_prelu(60)

        # pooling for classification
        self.pool1 = nn.AvgPool3d(kernel_size=(7, 7, 1), stride=(1, 1, 1))

        self.drop1 = nn.Dropout(0.5)

        self.fc1 = nn.Linear(in_features=60, out_features=classes)
        self.softm = nn.Softmax()

    def forward(self, x):
        # spectral conv layers
        x_spec0 = self.input_spec_conv(x)
        x_spec1 = self.spectral_conv1(x_spec0)
        x_spec1_ = torch.cat((x_spec0, x_spec1), dim=1)
        x_spec2 = self.spectral_conv2(x_spec1_)
        x_spec2_ = torch.cat((x_spec0, x_spec1, x_spec2), dim=1)
        x_spec3 = self.spectral_conv3(x_spec2_)
        xspec = torch.cat((x_spec0, x_spec1, x_spec2, x_spec3), dim=1)
        xspec = self.bn_prelu1(xspec)

        # transforming dims
        trans1 = self.reshape_conv(xspec)
        trans1 = self.bn_prelu2(trans1)
        trans1 = trans1.permute(0, 4, 2, 3, 1) # moving channels axis to the depth axis

        # spatial conv layers
        x_spat0 = self.input_spat_conv(trans1)
        x_spat1 = self.spatial_conv1(x_spat0)
        x_spat1_ = torch.cat((x_spat0, x_spat1), dim=1)
        x_spat2 = self.spatial_conv2(x_spat1_)
        x_spat2_ = torch.cat((x_spat0, x_spat1, x_spat2), dim=1)
        x_spat3 = self.spatial_conv3(x_spat2_)
        xspat = torch.cat((x_spat0, x_spat1, x_spat2, x_spat3), dim=1)
        xspat = self.bn_prelu3(xspat)

        pooled = self.pool1(xspat)
        flattened = torch.flatten(pooled, start_dim=1)
        dropped = self.drop1(flattened)
        logits = self.fc1(dropped)
        
        output = self.softm(logits)

        return output
   
class FerDSSC_model(nn.Module):

    def __init__(self, input_shape, classes):
        '''Input shape in the form of (C,H,W,D)'''
        super(FerDSSC_model, self).__init__()
        self.input_shape = input_shape

        global growth_rate
        growth_rate = 12

        # add padding since original paper would skip over channels that don't divide evenly into the convolution
        depth_after_spec = input_shape[3]
        # definitions for dense spectral layers
        self.input_spec_conv = nn.Conv3d(1, growth_rate * 2, kernel_size=(1, 1, 7), stride=(1, 1, 1), padding=(0, 0, 3))
        self.spectral_conv1 = Spectral_conv(growth_rate * 2)
        self.spectral_conv2 = Spectral_conv(growth_rate * 3)
        self.spectral_conv3 = Spectral_conv(growth_rate * 4)
        self.bn_prelu1 = Bn_prelu(growth_rate * 5)

        #reshaping and transforming to prep for spatial conv
        hidden_states = 30
        conv_depth = depth_after_spec # maybe consider setting this to an integer so we can apply transfer learning? (need to pad as well)
        self.reshape_conv = nn.Conv3d(growth_rate * 5, hidden_states, kernel_size=(1, 1, conv_depth), stride=(1, 1, 1))
        self.bn_prelu2 = Bn_prelu(hidden_states)

        # definitions for dense spatial layers
        self.input_spat_conv = nn.Conv3d(1, growth_rate * 2, kernel_size=(3, 3, hidden_states), stride=(1, 1, 1))
        self.spatial_conv1 = Spatial_conv(growth_rate * 2)
        self.spatial_conv2 = Spatial_conv(growth_rate * 3)
        self.spatial_conv3 = Spatial_conv(growth_rate * 4)
        self.bn_prelu3 = Bn_prelu(growth_rate * 5)

        # pooling for classification
        self.pool1 = nn.AvgPool3d(kernel_size=(7, 7, 1), stride=(1, 1, 1))

        self.drop1 = nn.Dropout(0.5)

        self.fc1 = nn.Linear(in_features=growth_rate * 5, out_features=classes)
        self.softm = nn.Softmax()

    def forward(self, x):
        # spectral conv layers
        x_spec0 = self.input_spec_conv(x)
        x_spec1 = self.spectral_conv1(x_spec0)
        x_spec1_ = torch.cat((x_spec0, x_spec1), dim=1)
        x_spec2 = self.spectral_conv2(x_spec1_)
        x_spec2_ = torch.cat((x_spec0, x_spec1, x_spec2), dim=1)
        x_spec3 = self.spectral_conv3(x_spec2_)
        xspec = torch.cat((x_spec0, x_spec1, x_spec2, x_spec3), dim=1)
        xspec = self.bn_prelu1(xspec)

        # transforming dims
        trans1 = self.reshape_conv(xspec)
        trans1 = self.bn_prelu2(trans1)
        trans1 = trans1.permute(0, 4, 2, 3, 1) # moving channels axis to the depth axis

        # spatial conv layers
        x_spat0 = self.input_spat_conv(trans1)
        x_spat1 = self.spatial_conv1(x_spat0)
        x_spat1_ = torch.cat((x_spat0, x_spat1), dim=1)
        x_spat2 = self.spatial_conv2(x_spat1_)
        x_spat2_ = torch.cat((x_spat0, x_spat1, x_spat2), dim=1)
        x_spat3 = self.spatial_conv3(x_spat2_)
        xspat = torch.cat((x_spat0, x_spat1, x_spat2, x_spat3), dim=1)
        xspat = self.bn_prelu3(xspat)

        pooled = self.pool1(xspat)
        flattened = torch.flatten(pooled, start_dim=1)
        dropped = self.drop1(flattened)
        logits = self.fc1(dropped)
        
        output = self.softm(logits)

        return output
    