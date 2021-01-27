import torch
import torch.nn as nn
import math
from model.subNets import *




class BridgeBlock(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(BridgeBlock, self).__init__()

        self.conv_bridge_1 = nn.Conv2d(in_channels=input_channel, out_channels=(input_channel*2 + output_channel)//3, kernel_size=3, stride=2, padding=1, bias=False)
        self.relu_bridge_1 = nn.PReLU()
        self.conv_bridge_2 = nn.Conv2d(in_channels=(input_channel*2 + output_channel)//3, out_channels=(input_channel + output_channel*2)//3, kernel_size=3, stride=2, padding=1, bias=False)
        self.relu_bridge_2 = nn.PReLU()
        self.conv_bridge_3 = nn.Conv2d(in_channels=(input_channel + output_channel*2)//3, out_channels=output_channel, kernel_size=3, stride=2, padding=1, bias=False)
        self.relu_bridge_3 = nn.PReLU()

    def forward(self, x):
        return self.relu_bridge_3(self.conv_bridge_3(
            self.relu_bridge_2(self.conv_bridge_2(
                self.relu_bridge_1(self.conv_bridge_1(x))
            ))
        ))





class GroupDesnseUnet(nn.Module):
    def __init__(self, nChannels=256, nDenselayers=8, growRate=64, nDenseBlocks=e):
        super(GroupDesnseUnet, self).__init__()

        self.grdb1 = GRDB(numofkernels=nChannels, nDenselayer=nDenselayers, growthRate=growRate, numforrg=nDenseBlocks)
        self.prelu1 = nn.PReLU()
        self.conv_down1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1, bias=False)
        self.grdb2 = GRDB(numofkernels=nChannels, nDenselayer=nDenselayers, growthRate=growRate, numforrg=nDenseBlocks)
        self.prelu2 = nn.PReLU()
        self.grdb3 = GRDB(numofkernels=nChannels, nDenselayer=nDenselayers, growthRate=growRate, numforrg=nDenseBlocks)




class MegaDenseUnet(nn.Module):
    def __init__(self, args):
        super(MegaDenseUnet, self).__init__()
        if args.jpeg_grid_add:
            input_channel = args.jpeg_grid_add
        else:
            input_channel = 3
        self.conv_input = nn.Conv2d(in_channels=input_channel, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu1 = nn.PReLU()
        self.down_conv = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False)
        self.relu2 = nn.PReLU()