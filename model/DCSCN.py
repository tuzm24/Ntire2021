import torch
import torch.nn as nn
from torchvision.models import densenet
from collections import OrderedDict
from torchsummary import summary
class Conv3x3(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, bias = False, activator=None,
                 use_batch_norm=False, drop_out=1.0):
        super(Conv3x3, self).__init__()
        if use_batch_norm:
            raise NotImplementedError
        self.add_module('3x3conv', nn.Conv2d(input_channel, output_channel, kernel_size, padding=(kernel_size//2), bias=bias))
        if activator is not None:
            self.add_module('PRELU', nn.PReLU())
        if drop_out != 1.0:
            self.add_module('dropout', nn.Dropout(p=drop_out))
    def forward(self, x):
        for name, layer in self.named_children():
            x = layer(x)
        return x

class SeveralLayers(nn.Module):
    def __init__(self, num_layers, input_channels, filters, min_filters, filters_decay_gamma, kernel_size=3, bias = False, activator='PRELU',
                 use_batch_norm=False, drop_out=1.0):
        super(SeveralLayers, self).__init__()
        output_channels = filters
        for i in range(num_layers):
            if i>0:
                x1 = i / float(num_layers - 1)
                y1 = pow(x1, 1.0 / filters_decay_gamma)
                output_channels = int((filters - min_filters) * (1 - y1) + min_filters)
            layer = Conv3x3(input_channel=input_channels, output_channel=output_channels, bias=bias, activator=activator,
                            use_batch_norm=use_batch_norm, kernel_size=kernel_size, drop_out=drop_out)
            input_channels = output_channels
            self.add_module('layer%d' % (i + 1), layer)

    def forward(self, x):
        features = []
        for name, layer in self.named_children():
            x = layer(x)
            features.append(x)
        return torch.cat(features, 1)



class DCSCN(nn.Module):
    def __init__(self, num_layers, recon_layers, input_channels, output_channels,
                 filters, min_filters, filters_decay_gamma, nin_filters, nin_filters2, kernel_size=3, bias = True, activator='PRELU',
                 use_batch_norm=False, drop_out=1.0, scale=2, reconstruct_filters=64):
        super(DCSCN, self).__init__()
        self.several_layers = SeveralLayers(num_layers, input_channels, filters, min_filters, filters_decay_gamma,
                                       kernel_size=kernel_size, bias=bias, activator=activator,
                                        use_batch_norm=use_batch_norm, drop_out=drop_out)
        total_output_feature_num = self.calcTotalOutputChannels(num_layers, filters, min_filters, filters_decay_gamma)
        self.A1 = Conv3x3(input_channel=total_output_feature_num, output_channel=nin_filters,
                          kernel_size=1, drop_out=drop_out, bias=bias, activator=activator)
        self.B1 = Conv3x3(input_channel=total_output_feature_num, output_channel=nin_filters2,
                          kernel_size=1, drop_out=drop_out, bias=bias, activator=activator)
        self.B2 = Conv3x3(input_channel=nin_filters2, output_channel=nin_filters2,
                          kernel_size=3, drop_out=drop_out, bias=bias, activator=activator)
        self.upsampling_conv =  Conv3x3(input_channel=(nin_filters + nin_filters2), output_channel=(nin_filters + nin_filters2) * scale * scale,
                                                             kernel_size=kernel_size, use_batch_norm=False, bias=True)
        self.pixelshuffler =  nn.PixelShuffle(2)
        input_recon_channel = (nin_filters + nin_filters2)
        self.recon_subnet = nn.Sequential()
        for i in range(recon_layers-1):
            self.recon_subnet.add_module('recon_cnn {}'.format(i), Conv3x3(input_recon_channel, reconstruct_filters, kernel_size=kernel_size,
                                                                           drop_out=drop_out, activator=activator))
            input_recon_channel = reconstruct_filters
        self.lastconv = Conv3x3(input_recon_channel, output_channels, kernel_size=kernel_size)


    def calcTotalOutputChannels(self, num_layers, filters, min_filters, filters_decay_gamma):
        output_channels = filters
        total_output_feature_num = 0
        for i in range(num_layers):
            if i>0:
                x1 = i / float(num_layers - 1)
                y1 = pow(x1, 1.0 / filters_decay_gamma)
                output_channels = int((filters - min_filters) * (1 - y1) + min_filters)

            total_output_feature_num += output_channels
        return total_output_feature_num

    def forward(self,x):
        x = self.several_layers(x)
        a_x = self.A1(x)
        b_x = self.B2(self.B1(x))
        x = torch.cat([a_x, b_x], 1)
        x = self.pixelshuffler(self.upsampling_conv(x))
        return self.lastconv(self.recon_subnet(x))

if __name__=='__main__':
    model = DCSCN(num_layers=7, filters=32, min_filters=8, filters_decay_gamma=1.2, nin_filters=24, nin_filters2=8, recon_layers=0,
          input_channels=1, output_channels=1)
    print(model)
    model(torch.randn((2,1,32,32)))
