import torch
import torch.nn as nn
from help_torch import RgbToYcbcr
from help_torch import YcbcrToRgb
from model.cbam import CBAM
def make_model(args, parent=False):
    return _NetG(args)

class _Residual_Block(nn.Module):
    def __init__(self, input_dim=256):
        super(_Residual_Block, self).__init__()

        #res1
        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu2 = nn.PReLU()
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu4 = nn.PReLU()
        #res1
        #concat1

        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1, bias=False)
        self.relu6 = nn.PReLU()

        #res2
        self.conv7 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu8 = nn.PReLU()
        #res2
        #concat2

        self.conv9 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=1, bias=False)
        self.relu10 = nn.PReLU()

        #res3
        self.conv11 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu12 = nn.PReLU()
        #res3

        self.conv13 = nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=1, stride=1, padding=0, bias=False)
        self.up14 = nn.PixelShuffle(2)

        #concat2
        self.conv15 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False)
        #res4
        self.conv16 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu17 = nn.PReLU()
        #res4

        self.conv18 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.up19 = nn.PixelShuffle(2)

        #concat1
        self.conv20 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False)
        #res5
        self.conv21 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu22 = nn.PReLU()
        self.conv23 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu24 = nn.PReLU()
        #res5

        self.conv25 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)


    def forward(self, x):
        res1 = x
        out = self.relu4(self.conv3(self.relu2(self.conv1(x))))
        out = torch.add(res1, out)
        cat1 = out

        out = self.relu6(self.conv5(out))
        res2 = out
        out = self.relu8(self.conv7(out))
        out = torch.add(res2, out)
        cat2 = out

        out = self.relu10(self.conv9(out))
        res3 = out

        out = self.relu12(self.conv11(out))
        out = torch.add(res3, out)

        out = self.up14(self.conv13(out))

        out = torch.cat([out, cat2], 1)
        out = self.conv15(out)
        res4 = out
        out = self.relu17(self.conv16(out))
        out = torch.add(res4, out)

        out = self.up19(self.conv18(out))

        out = torch.cat([out, cat1], 1)
        out = self.conv20(out)
        res5 = out
        out = self.relu24(self.conv23(self.relu22(self.conv21(out))))
        out = torch.add(res5, out)

        out = self.conv25(out)
        out = torch.add(out, res1)

        return out

class Recon_Block(nn.Module):
    def __init__(self):
        super(Recon_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu2 = nn.PReLU()
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu4 = nn.PReLU()

        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu6= nn.PReLU()
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu8 = nn.PReLU()

        self.conv9 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu10 = nn.PReLU()
        self.conv11 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu12 = nn.PReLU()

        self.conv13 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu14 = nn.PReLU()
        self.conv15 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu16 = nn.PReLU()

        self.conv17 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)


    def forward(self, x):
        res1 = x
        output = self.relu4(self.conv3(self.relu2(self.conv1(x))))
        output = torch.add(output, res1)

        res2 = output
        output = self.relu8(self.conv7(self.relu6(self.conv5(output))))
        output = torch.add(output, res2)

        res3 = output
        output = self.relu12(self.conv11(self.relu10(self.conv9(output))))
        output = torch.add(output, res3)

        res4 = output
        output = self.relu16(self.conv15(self.relu14(self.conv13(output))))
        output = torch.add(output, res4)

        output = self.conv17(output)
        output = torch.add(output, res1)

        return output



class _NetG(nn.Module):
    def __init__(self,args):
        super(_NetG, self).__init__()
        self.isjpg = args.jpeg_grid_add
        if args.jpeg_grid_add:
            input_channel = args.jpeg_grid_add
        else:
            input_channel = 3
        self.rgb_to_yuv = RgbToYcbcr()


        self.yconv = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        )

        self.ybam = CBAM(128)
        self.cconv = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.cbam = CBAM(128)
        self.cdown = nn.Conv2d(in_channels=257, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False)



        self.recursive_A = _Residual_Block(input_dim=257)
        self.recursive_B = _Residual_Block()
        self.recursive_C = _Residual_Block()
        self.recursive_D = _Residual_Block()
        self.recursive_E = _Residual_Block(input_dim=257)
        self.recursive_F = _Residual_Block()


        self.yrecon = Recon_Block()
        self.crecon = Recon_Block()
        #concat

        self.y_mid = nn.Sequential(
            nn.Conv2d(in_channels=256*4, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.PReLU()
        )
        self.y_mid2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PReLU()
        )
        self.c_mid = nn.Sequential(
            nn.Conv2d(in_channels=256*2, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.PReLU()
        )
        self.c_mid2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PReLU()
        )



        self.yconv_output = nn.Sequential(nn.PixelShuffle(2),
                                          nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False))
        self.cconv_output = nn.Sequential(nn.PixelShuffle(2),
                                          nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False),
                                          nn.PixelShuffle(2),
                                          nn.Conv2d(in_channels=16, out_channels=2, kernel_size=3, stride=1, padding=1,
                                                    bias=False)
        )
        self.yuv_to_rgb = YcbcrToRgb()




    def forward(self, x):
        yuv_input = self.rgb_to_yuv(x[:,:3,...])
        grid = x[:,3:,...]
        residual = yuv_input
        y_out = self.yconv(torch.cat([yuv_input[:,:1,...], grid], dim=1))
        c_out = self.cconv(yuv_input[:,1:3,...])

        y_out = torch.cat([y_out, self.cbam(c_out)], dim=1)
        c_out = torch.cat([c_out, self.ybam(y_out), grid], dim=1)
        c_out = torch.cat([self.cdown(c_out)], dim=1)


        out1 = self.recursive_A(y_out)
        out2 = self.recursive_B(out1)
        out3 = self.recursive_C(out2)
        out4 = self.recursive_D(out3)
        out5 = self.recursive_E(c_out)
        out6 = self.recursive_F(out5)

        recon1 = self.recon(out1)
        recon2 = self.recon(out2)
        recon3 = self.recon(out3)
        recon4 = self.recon(out4)
        recon5 = self.recon(out5)
        recon6 = self.recon(out6)

        y_out = torch.cat([recon1, recon2, recon3, recon4], 1)
        c_out = torch.cat([recon5, recon6], 1)


        y_out = self.y_mid(y_out)
        y_res = y_out
        y_out = self.y_mid2(y_out)
        y_out = torch.add(y_out, y_res)
        y_out = torch.add(self.yconv_output(y_out), residual[:,:1,...])

        c_out = self.c_mid(c_out)
        c_res = c_out
        c_out = self.c_mid2(c_out)
        c_out = torch.add(c_out, c_res)
        c_out = torch.add(self.cconv_output(c_out), residual[:,1:3,...])



        out = self.yuv_to_rgb(torch.cat([y_out,c_out], dim=1))
        return out