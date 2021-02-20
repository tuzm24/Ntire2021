import torch
import torch.nn as nn
from help_torch import RgbToYcbcr
from help_torch import YcbcrToRgb


def make_model(args, parent=False):
    # return _UUU(args)
    pass


class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)


class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G = growRate
        C = nConvLayers

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c * G, G))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C * G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x

class _Base_UNet(nn.Module):
    def __init__(self, in_channels=256, out_channels=256, base_channels=256):
        super(_Base_UNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=base_channels, kernel_size=3,
                               padding=1, stride=1, bias=False)
        self.relu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(in_channels=base_channels + in_channels, out_channels=base_channels, kernel_size=3,
                               padding=1, stride=1, bias=False)
        self.relu2 = nn.PReLU()


        self.conv3 = nn.Conv2d(in_channels=base_channels*2 + in_channels, out_channels=base_channels*2, kernel_size=3,
                               padding=1, stride=2, bias=False)
        self.relu3 = nn.PReLU()
        self.conv4 = nn.Conv2d(in_channels=base_channels*2, out_channels=base_channels*2, kernel_size=3,
                               padding=1, stride=1, bias=False)
        self.relu4 = nn.PReLU()

        self.conv5 = nn.Conv2d(in_channels=base_channels*4, out_channels=base_channels*4, kernel_size=3,
                               padding=1, stride=2, bias=False)
        self.relu5 = nn.PReLU()
        self.conv6 = nn.Conv2d(in_channels=base_channels*4, out_channels=base_channels*4, kernel_size=3,
                               padding=1, stride=1, bias=False)
        self.relu6 = nn.PReLU()
        self.conv7 = nn.Conv2d(in_channels=base_channels*8, out_channels=base_channels*4, kernel_size=3,
                               padding=1, stride=1, bias=False)
        self.relu7 = nn.PReLU()

        self.conv_up1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1, bias=False)

        self.conv8 = nn.Conv2d(in_channels=base_channels * 2, out_channels=base_channels * 2, kernel_size=3,
                               padding=1, stride=1, bias=False)
        self.relu8 = nn.PReLU()
        self.conv9 = nn.Conv2d(in_channels=base_channels * 4, out_channels=base_channels * 2, kernel_size=3,
                               padding=1, stride=1, bias=False)
        self.relu9 = nn.PReLU()

        self.conv_up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1, bias=False)

        self.conv10 = nn.Conv2d(in_channels=base_channels, out_channels=base_channels, kernel_size=3,
                               padding=1, stride=1, bias=False)
        self.relu10 = nn.PReLU()
        self.conv11 = nn.Conv2d(in_channels=base_channels*2, out_channels=base_channels, kernel_size=3,
                               padding=1, stride=1, bias=False)
        self.relu11 = nn.PReLU()

        self.conv12 = nn.Conv2d(in_channels=base_channels, out_channels=out_channels, kernel_size=3,
                               padding=1, stride=1, bias=False)

    def forward(self, x):
        out0 = self.relu1(self.conv1(x))
        out = torch.cat([x, out0], dim=1)
        out1 = self.relu2(self.conv2(out))
        out = torch.cat([out, out1], dim=1)

        out = self.relu3(self.conv3(out))
        out2 = self.relu4(self.conv4(out))
        out = torch.cat([out, out2], dim=1)

        out = self.relu5(self.conv5(out))
        out3 = self.relu6(self.conv6(out))
        out = torch.cat([out, out3], dim=1)
        out = self.conv_up1(self.relu7(self.conv7(out)))

        out = torch.add(out2, out)
        out2 = self.relu8(self.conv8(out))
        out = torch.cat([out2, out], dim=1)
        out = self.conv_up2(self.relu9(self.conv9(out)))

        out = torch.add(out1, out)
        out1 = self.relu10(self.conv10(out))
        out = torch.cat([out, out1], dim=1)
        return torch.add(out0, self.conv12(self.relu11(self.conv11(out))))

class _UUUU(nn.Module):
    def __init__(self, args):
        super(_UUUU, self).__init__()
        self.isjpg = 3
        # self.isjpg = args.jpeg_grid_add
        if self.isjpg:
            input_channel = self.isjpg
        else:
            input_channel = 3
        self.rgb_to_yuv = RgbToYcbcr()
        self.conv_input = nn.Conv2d(in_channels=input_channel, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu1 = nn.PReLU()
        self.conv_down = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False)
        self.relu2 = nn.PReLU()

        self.Unet1 = _Base_UNet()
        self.Unet2 = _Base_UNet(in_channels=256*2)
        self.Unet3 = _Base_UNet(in_channels=256*3)
        self.Unet4 = _Base_UNet(in_channels=256*4)
        self.Unet5 = _Base_UNet(in_channels=256*5)
        self.Unet6 = _Base_UNet(in_channels=256*6)

        self.conv_bottle = nn.Conv2d(in_channels=256*7, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False)

        self.num_dense = 12
        self.RDBs = nn.ModuleList()
        for i in range(self.num_dense):
            self.RDBs.append(
                RDB(growRate0=256, growRate=64, nConvLayers=8)
            )


        self.conv_mid = nn.Sequential(*[
            nn.Conv2d(self.num_dense*256, 256, 1, padding=0, stride=1),
            nn.Conv2d(256,256, kernel_size=3, padding=1, stride=1)
        ])

        self.upsampling = nn.PixelShuffle(2)

        self.conv_output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)

        self.yuv_to_rgb = YcbcrToRgb()

    def forward(self, x):
        yuv_input = self.rgb_to_yuv(x[:,:3,...])
        residual = yuv_input
        if self.isjpg:
            _input = torch.cat([yuv_input, x[:,3:,...]], dim=1)
        else:
            _input = yuv_input
        out = self.relu1(self.conv_input(_input))
        out = self.relu2(self.conv_down(out))

        out1 = self.Unet1(out)
        out = torch.cat([out, out1], dim=1)
        out1 = self.Unet2(out)
        out = torch.cat([out, out1], dim=1)
        out1 = self.Unet3(out)
        out = torch.cat([out, out1], dim=1)
        out1 = self.Unet4(out)
        out = torch.cat([out, out1], dim=1)
        out1 = self.Unet5(out)
        out = torch.cat([out, out1], dim=1)
        out1 = self.Unet6(out)
        out = torch.cat([out, out1], dim=1)

        out = self.conv_bottle(out)

        RDBs_out = []
        for i in range(self.num_dense):
            out = self.RDBs[i](out)
            RDBs_out.append(out)


        out = self.conv_mid(torch.cat(RDBs_out, dim=1))
        out = self.upsampling(out)
        out = self.conv_output(out)
        out = torch.add(out, residual)

        out = self.yuv_to_rgb(out)
        return out




if __name__ == '__main__':
    from my_torchsummary import summary

    m = _UUUU(1).cuda()

    # inpt = torch.randn((1,5,96,96)).cuda()
    # while True:
    #     m(inpt)
    torch.set_grad_enabled(False)
    m.eval()
    summary(m, (3, 1280, 720))







