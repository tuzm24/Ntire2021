import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class make_dense(nn.Module):
    def __init__(self, nChannels, nChannels_, growthRate, kernel_size=3):
        super(make_dense, self).__init__()
        self.conv = nn.Conv2d(nChannels_, growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                              bias=False)
        self.prelu = nn.PReLU()
        self.nChannels = nChannels

    def forward(self, x):
        out = self.prelu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


# Residual dense block (RDB) architecture
class RDB(nn.Module):
    """
    https://github.com/lizhengwei1992/ResidualDenseNetwork-Pytorch
    """

    def __init__(self, nChannels, nDenselayer, growthRate):
        """
        :param nChannels: input feature 의 channel 수
        :param nDenselayer: RDB(residual dense block) 에서 Conv 의 개수
        :param growthRate: Conv 의 output layer 의 수
        """
        super(RDB, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):
            modules.append(make_dense(nChannels, nChannels_, growthRate))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)

        ###################kingrdb ver2##############################################
        # self.conv_1x1 = nn.Conv2d(nChannels_ + growthRate, nChannels, kernel_size=1, padding=0, bias=False)
        ###################else######################################################
        self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        # local residual 구조
        out = out + x
        return out

class GRDB(nn.Module):
    """
    https://github.com/lizhengwei1992/ResidualDenseNetwork-Pytorch
    """

    def __init__(self, numofkernels, nDenselayer, growthRate, numdb):
        """
        :param nChannels: input feature 의 channel 수
        :param nDenselayer: RDB(residual dense block) 에서 Conv 의 개수
        :param growthRate: Conv 의 output layer 의 수
        """
        super(GRDB, self).__init__()

        modules = []
        for i in range(numdb):
            modules.append(RDB(numofkernels, nDenselayer=nDenselayer, growthRate=growthRate))
        self.rdbs = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(numofkernels * numdb, numofkernels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = x
        outputlist = []
        for rdb in self.rdbs:
            output = rdb(out)
            outputlist.append(output)
            out = output
        concat = torch.cat(outputlist, 1)
        out = x + self.conv_1x1(concat)
        return out

class BridgeBlock(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(BridgeBlock, self).__init__()

        self.conv_bridge_1 = nn.Conv2d(in_channels=input_channel, out_channels=(input_channel*2 + output_channel)//3, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu_bridge_1 = nn.PReLU()
        self.conv_bridge_2 = nn.Conv2d(in_channels=(input_channel*2 + output_channel)//3, out_channels=(input_channel + output_channel*2)//3, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu_bridge_2 = nn.PReLU()
        self.conv_bridge_3 = nn.Conv2d(in_channels=(input_channel + output_channel*2)//3, out_channels=output_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu_bridge_3 = nn.PReLU()

    def forward(self, x):
        return self.relu_bridge_3(self.conv_bridge_3(
            self.relu_bridge_2(self.conv_bridge_2(
                self.relu_bridge_1(self.conv_bridge_1(x))
            ))
        ))





class GroupDesnseUnet(nn.Module):
    def __init__(self, nChannels=128, nDenselayers=4, growRate=64, nDenseBlocks=4):
        super(GroupDesnseUnet, self).__init__()
        self.grdb1 = GRDB(numofkernels=nChannels, nDenselayer=nDenselayers, growthRate=growRate, numdb=nDenseBlocks)
        self.conv_down1 = nn.Conv2d(in_channels=nChannels, out_channels=nChannels*2, kernel_size=3, stride=2, padding=1, bias=False)
        self.prelu1 = nn.PReLU()
        self.grdb2 = GRDB(numofkernels=nChannels*2, nDenselayer=nDenselayers, growthRate=growRate, numdb=nDenseBlocks)
        self.conv_down2 = nn.Conv2d(in_channels=nChannels*2, out_channels=nChannels*4, kernel_size=3, stride=2, padding=1, bias=False)
        self.prelu2 = nn.PReLU()
        self.grdb3 = GRDB(numofkernels=nChannels*4, nDenselayer=nDenselayers, growthRate=growRate, numdb=nDenseBlocks)
        self.conv_up1 = nn.ConvTranspose2d(nChannels*4, nChannels*2, kernel_size=4, stride=2, padding=1, bias=False)

        self.prelu3 = nn.PReLU()
        self.grdb4 = GRDB(numofkernels=nChannels*2, nDenselayer=nDenselayers, growthRate=growRate, numdb=nDenseBlocks)
        self.conv_up2 = nn.ConvTranspose2d(nChannels*2, nChannels, kernel_size=4, stride=2, padding=1, bias=False)

        self.prelu4 = nn.PReLU()
        self.grdb5 = GRDB(numofkernels=nChannels, nDenselayer=nDenselayers, growthRate=growRate, numdb=nDenseBlocks)

        self.bridge1 = BridgeBlock(nChannels, nChannels)
        self.bridge2 = BridgeBlock(nChannels*2 , nChannels *2)

    def forward(self, x):
        x = self.grdb1(x)
        x1 = self.bridge1(x)
        x = self.prelu1(self.conv_down1(x))
        x = self.grdb2(x)
        x2 = self.bridge2(x)
        x = self.prelu2(self.conv_down2(x))
        x = self.grdb3(x)
        x = self.prelu3(self.conv_up1(x))
        x = self.grdb4(x + x2)
        x = self.prelu4(self.conv_up2(x))
        x = self.grdb5(x + x1)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=('avg', 'max')):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

class GridDenseChannelExt(nn.Module):
    def __init__(self, in_channels, out_channels=256):

        super(GridDenseChannelExt, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride =1, padding=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(in_channels=64, out_channels=(128+out_channels)//3, kernel_size=3, stride =1, padding=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(in_channels=(128+out_channels)//3, out_channels=out_channels, kernel_size=3, stride =1, padding=1, bias=False),
            ChannelGate(gate_channels=out_channels)
        )
    def forward(self, x):
        return self.convs(x)






class MegaDenseUnet(nn.Module):
    def __init__(self, args):
        super(MegaDenseUnet, self).__init__()
        self.base_channel = 128
        self.nDenselayers = 4
        self.growRate = 64
        self.nDenseBlocks = 4

        self.gdce1 = GridDenseChannelExt(in_channels=1, out_channels=self.base_channel)
        self.gdce2 = GridDenseChannelExt(in_channels=1, out_channels=self.base_channel)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.base_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(in_channels=self.base_channel, out_channels=self.base_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.prelu2 = nn.PReLU()

        self.gdu1 = GroupDesnseUnet(nChannels=self.base_channel, nDenselayers=self.nDenselayers,
                                    nDenseBlocks=self.nDenseBlocks, growRate=self.growRate)
        self.gdu2 = GroupDesnseUnet(nChannels=self.base_channel, nDenselayers=self.nDenselayers,
                                    nDenseBlocks=self.nDenseBlocks, growRate=self.growRate)
        self.gdu3 = GroupDesnseUnet(nChannels=self.base_channel, nDenselayers=self.nDenselayers,
                                    nDenseBlocks=self.nDenseBlocks, growRate=self.growRate)
        self.gdu4 = GroupDesnseUnet(nChannels=self.base_channel, nDenselayers=self.nDenselayers,
                                    nDenseBlocks=self.nDenseBlocks, growRate=self.growRate)

        self.recon = nn.Conv2d(in_channels=self.base_channel * 4, out_channels=self.base_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.prelu3 = nn.PReLU()
        self.gdu5 = GroupDesnseUnet(nChannels=self.base_channel, nDenselayers=self.nDenselayers,
                                    nDenseBlocks=self.nDenseBlocks, growRate=self.growRate)
        self.finalconv = nn.Conv2d(in_channels=self.base_channel, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self,x ):
        residual = x[:,:3,...]
        grid8x8 = self.gdce1(x[:,3:4,...])
        grid16x16 = self.gdce2(x[:,4:5,...])
        x = self.prelu2(self.conv2(self.prelu1(self.conv1(x[:,:3,...]))))
        x1 = self.gdu1(x)
        x2 = self.gdu2(x)
        x3 = self.gdu3(x + grid8x8)
        x4 = self.gdu4(x + grid16x16)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.prelu3(self.recon(x))
        x = self.gdu5(x)
        return self.finalconv(x) + residual






if __name__ == '__main__':
    from my_torchsummary import summary

    m = MegaDenseUnet(1).cuda()
    inpt = torch.randn((2,5,96,96)).cuda()

    while True:
        m(inpt)
    torch.set_grad_enabled(False)
    m.eval()
    summary(m, (5, 64, 64))