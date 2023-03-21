import torch
import torch.nn as nn


class UNetDoubleConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1, 
                 stride: int = 1, dilation: int = 1, padding: int = 0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
                      padding=padding, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size,
                      padding=padding, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        return out


class UNetDoubleDeconv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1, 
                 stride: int = 1, dilation: int = 1, padding: int = 0, output_padding: int = 0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, 
                               padding=padding, dilation=dilation, output_padding=output_padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, 
                      padding=padding, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        return out


class SelfAttention(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.query_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, width, height = x.size()
        query = self.query_conv(x).view(batch_size, -1, width * height).transpose(1, 2)
        key = self.key_conv(x).view(batch_size, -1, width * height)
        value = self.value_conv(x).view(batch_size, -1, width * height)
        energy = torch.bmm(query, key)
        attention = self.softmax(energy)
        out = torch.bmm(value, attention.transpose(1, 2))
        out = out.view(batch_size, channels, width, height)
        out = self.gamma * out + x
        return out


class DSA_UNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.in_conv = nn.Conv2d(input_dim, 32, kernel_size=1)
        self.down1 = UNetDoubleConv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.down2 = UNetDoubleConv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.down3 = UNetDoubleConv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.down4 = UNetDoubleConv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.dilated_conv1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.dilated_conv2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, dilation=2, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.dilated_conv3 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, dilation=4, padding=4),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.dilated_conv4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, dilation=8, padding=8),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.up4 = UNetDoubleDeconv2d(1024, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.attn4 = SelfAttention(256)
        self.up3 = UNetDoubleDeconv2d(512, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.attn3 = SelfAttention(128)
        self.up2 = UNetDoubleDeconv2d(256, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up1 = UNetDoubleDeconv2d(128, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.in_conv(x)
        h2 = self.down1(h1)
        h3 = self.down2(h2)
        h4 = self.down3(h3)
        h5 = self.down4(h4)
        hd = self.dilated_conv1(h5)
        hd = self.dilated_conv2(hd)
        hd = self.dilated_conv3(hd)
        hd = self.dilated_conv4(hd)
        h4p = self.up4(torch.cat([hd, h5], dim=1))
        h4p = self.attn4(h4p)
        h3p = self.up3(torch.cat([h4p, h4], dim=1))
        h3p = self.attn3(h3p)
        h2p = self.up2(torch.cat([h3p, h3], dim=1))
        h1p = self.up1(torch.cat([h2p, h2], dim=1))
        out = self.out_conv(torch.cat([h1p, h1], dim=1))
        out = self.act(out)
        return out
