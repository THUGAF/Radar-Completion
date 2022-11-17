import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import *


# refers to https://github.com/otenim/GLCIC-PyTorch
class GLCIC(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # input_shape: (None, input_dim, img_h, img_w)
        self.conv1 = nn.Conv2d(input_dim, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU()
        # input_shape: (None, 64, img_h, img_w)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.act2 = nn.ReLU()
        # input_shape: (None, 128, img_h//2, img_w//2)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.act3 = nn.ReLU()
        # input_shape: (None, 128, img_h//2, img_w//2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.act4 = nn.ReLU()
        # input_shape: (None, 256, img_h//4, img_w//4)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.act5 = nn.ReLU()
        # input_shape: (None, 256, img_h//4, img_w//4)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.act6 = nn.ReLU()
        # input_shape: (None, 256, img_h//4, img_w//4)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=1, dilation=2, padding=2)
        self.bn7 = nn.BatchNorm2d(256)
        self.act7 = nn.ReLU()
        # input_shape: (None, 256, img_h//4, img_w//4)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, stride=1, dilation=4, padding=4)
        self.bn8 = nn.BatchNorm2d(256)
        self.act8 = nn.ReLU()
        # input_shape: (None, 256, img_h//4, img_w//4)
        self.conv9 = nn.Conv2d(256, 256, kernel_size=3, stride=1, dilation=8, padding=8)
        self.bn9 = nn.BatchNorm2d(256)
        self.act9 = nn.ReLU()
        # input_shape: (None, 256, img_h//4, img_w//4)
        self.conv10 = nn.Conv2d(256, 256, kernel_size=3, stride=1, dilation=16, padding=16)
        self.bn10 = nn.BatchNorm2d(256)
        self.act10 = nn.ReLU()
        # input_shape: (None, 256, img_h//4, img_w//4)
        self.conv11 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn11 = nn.BatchNorm2d(256)
        self.act11 = nn.ReLU()
        # input_shape: (None, 256, img_h//4, img_w//4)
        self.conv12 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn12 = nn.BatchNorm2d(256)
        self.act12 = nn.ReLU()
        # input_shape: (None, 256, img_h//4, img_w//4)
        self.deconv13 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn13 = nn.BatchNorm2d(128)
        self.act13 = nn.ReLU()
        # input_shape: (None, 128, img_h//2, img_w//2)
        self.conv14 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn14 = nn.BatchNorm2d(128)
        self.act14 = nn.ReLU()
        # input_shape: (None, 128, img_h//2, img_w//2)
        self.deconv15 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn15 = nn.BatchNorm2d(64)
        self.act15 = nn.ReLU()
        # input_shape: (None, 64, img_h, img_w)
        self.conv16 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.bn16 = nn.BatchNorm2d(32)
        self.act16 = nn.ReLU()
        # input_shape: (None, 32, img_h, img_w)
        self.conv17 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        self.act17 = nn.Sigmoid()
        # output_shape: (None, 1, img_h, img_w)

    def forward(self, x):
        x = self.bn1(self.act1(self.conv1(x)))
        x = self.bn2(self.act2(self.conv2(x)))
        x = self.bn3(self.act3(self.conv3(x)))
        x = self.bn4(self.act4(self.conv4(x)))
        x = self.bn5(self.act5(self.conv5(x)))
        x = self.bn6(self.act6(self.conv6(x)))
        x = self.bn7(self.act7(self.conv7(x)))
        x = self.bn8(self.act8(self.conv8(x)))
        x = self.bn9(self.act9(self.conv9(x)))
        x = self.bn10(self.act10(self.conv10(x)))
        x = self.bn11(self.act11(self.conv11(x)))
        x = self.bn12(self.act12(self.conv12(x)))
        x = self.bn13(self.act13(self.deconv13(x)))
        x = self.bn14(self.act14(self.conv14(x)))
        x = self.bn15(self.act15(self.deconv15(x)))
        x = self.bn16(self.act16(self.conv16(x)))
        x = self.act17(self.conv17(x))
        return x


class DilatedUNet(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        # input_shape: (None, in_channels, img_h, img_w)
        self.in_conv = nn.Conv2d(in_channels, 64, kernel_size=1)
        # input_shape: (None, 64, img_h, img_w)
        self.down1 = DoubleConv2d(64, 128, kernel_size=3, stride=2, padding=1)
        # input_shape: (None, 128, img_h//2, img_w//2)
        self.down2 = DoubleConv2d(128, 256, kernel_size=3, stride=2, padding=1)
        # input_shape: (None, 256, img_h//4, img_w//4)
        self.dilated_conv1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        # input_shape: (None, 256, img_h//4, img_w//4)
        self.dilated_conv2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, dilation=2, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        # input_shape: (None, 256, img_h//4, img_w//4)
        self.dilated_conv3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, dilation=4, padding=4),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        # input_shape: (None, 256, img_h//4, img_w//4)
        self.dilated_conv4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, dilation=8, padding=8),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        # input_shape: (None, 256, img_h//4, img_w//4)
        self.up2 = DoubleDeconv2d(512, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        # input_shape: (None, 128, img_h//2, img_w//2)
        self.up1 = DoubleDeconv2d(256, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        # input_shape: (None, 64, img_h, img_w)
        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.act = nn.Sigmoid()
        # output_shape: (None, 1, img_h, img_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_conv(x)
        h1 = self.down1(x)
        h2 = self.down2(h1)
        hd = self.dilated_conv1(h2)
        hd = self.dilated_conv2(hd)
        hd = self.dilated_conv3(hd)
        hd = self.dilated_conv4(hd)
        h2p = self.up2(torch.cat([hd, h2], dim=1))
        h1p = self.up1(torch.cat([h2p, h1], dim=1))
        out = self.out_conv(h1p)
        out = self.act(out)
        return out


class UNet(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        # Downscaling
        self.in_conv = nn.Conv2d(in_channels, 64, kernel_size=1)
        self.downscaler = nn.MaxPool2d(2, 2)
        self.down1 = DoubleConv2d(64, 128)
        self.down2 = DoubleConv2d(128, 256)
        self.down3 = DoubleConv2d(256, 256)
        # Upscaling
        self.upscaler = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up3 = DoubleConv2d(512, 128)
        self.up2 = DoubleConv2d(256, 64)
        self.up1 = DoubleConv2d(128, 64)
        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Downscaling
        h1 = self.in_conv(x)
        h2 = self.down1(self.downscaler(h1))
        h3 = self.down2(self.downscaler(h2))
        h4 = self.down3(self.downscaler(h3))
        # Upscaling
        h3p = self.up3(torch.cat([self.upscaler(h4), h3], dim=1))
        h2p = self.up2(torch.cat([self.upscaler(h3p), h2], dim=1))
        h1p = self.up1(torch.cat([self.upscaler(h2p), h1], dim=1))
        out = self.out_conv(h1p)
        out = self.act(out)
        return out


class UNet_SA(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        # Downscaling
        self.in_conv = nn.Conv2d(in_channels, 64, kernel_size=1)
        self.downscaler = nn.MaxPool2d(2, 2)
        self.down1 = DoubleConv2d(64, 128)
        self.down2 = DoubleConv2d(128, 256)
        self.down3 = DoubleConv2d(256, 256)
        # Upscaling
        self.upscaler = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up3 = DoubleConv2d(512, 128)
        self.self_attention2 = SelfAttention(128)
        self.up2 = DoubleConv2d(256, 64)
        self.self_attention1 = SelfAttention(64)
        self.up1 = DoubleConv2d(128, 64)
        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Downscaling
        h1 = self.in_conv(x)
        h2 = self.down1(self.downscaler(h1))
        h3 = self.down2(self.downscaler(h2))
        h4 = self.down3(self.downscaler(h3))
        # Upscaling
        h3p = self.up3(torch.cat([self.upscaler(h4), h3], dim=1))
        h3p = self.self_attention2(h3p)
        h2p = self.up2(torch.cat([self.upscaler(h3p), h2], dim=1))
        h2p = self.self_attention1(h2p)
        h1p = self.up1(torch.cat([self.upscaler(h2p), h1], dim=1))
        out = self.out_conv(h1p)
        out = self.act(out)
        return out
