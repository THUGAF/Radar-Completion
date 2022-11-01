import torch
import torch.nn as nn


class DoubleConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super(DoubleConv2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        return out


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super(Down, self).__init__()
        self.downscaler = nn.MaxPool2d(2, 2)
        self.conv = DoubleConv2d(in_channels, out_channels, kernel_size, padding)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downscaler(x)
        out = self.conv(x)
        return out


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super(Up, self).__init__()
        self.upscaler = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv = DoubleConv2d(in_channels, out_channels, kernel_size, padding)
    
    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        x = self.upscaler(x)
        out = torch.cat([x, h], dim=1)
        out = self.conv(out)
        return out