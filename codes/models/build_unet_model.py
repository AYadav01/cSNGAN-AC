import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool3d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()
        if bilinear:
            # self.up = nn.UpsamplingBilinear2d(scale_factor=2)
            self.up = nn.Upsample(scale_factor=2)
        else:
            self.up = nn.ConvTranspose3d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x
        

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf):
        super(UNet, self).__init__()
        self.inc = inconv(in_nc, nf)
        self.down1 = down(nf, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, nf)
        self.up4 = up(128, nf)
        self.outc = outconv(nf, out_nc)
        self.Dropout = nn.Dropout(0.2)

    def forward(self, x):
        # print('-'*40)
        # print('incoming x shape:', x.shape)
        x1 = self.inc(x)
        # print('shape after first inc:', x1.shape)
        x2 = self.down1(x1)
        # print('shape after first down1:', x2.shape)
        x3 = self.down2(x2)
        # print('shape after first down2:', x3.shape)
        x3 = self.Dropout(x3)
        # print('shape after first dropout:', x3.shape)
        x4 = self.down3(x3)
        # print('shape after first down3:', x4.shape)
        x5 = self.down4(x4)
        # print('shape after first down4:', x5.shape)
        x = self.up1(x5, x4)
        # print('shape after first up1:', x.shape)
        x = self.up2(x, x3)
        # print('shape after first up2:', x.shape)
        x = self.up3(x, x2)
        # print('shape after first up3:', x.shape)
        x = self.up4(x, x1)
        # print('shape after first up4:', x.shape)
        x = self.outc(x)
        # print('final output:', x.shape)
        return x