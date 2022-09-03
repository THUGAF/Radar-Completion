from typing import Tuple, Union
import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """ConvLSTM Cell without peephole connection.
        
    Args:
        channels (int): number of input channels.
        filters (int): number of convolutional kernels.
        kernel_size (int, tuple): size of convolutional kernels.
        padding (int, tuple): size of padding.
    """
    
    def __init__(self, channels: int, filters: int, kernel_size: Union[int, tuple] = 3, 
                 padding: Union[int, tuple] = 1):
        super(ConvLSTMCell, self).__init__()
        self.filters = filters
        self.conv = nn.Conv2d(channels + filters, filters * 4, kernel_size, padding=padding)
        
    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: 4D tensor (B, C, H, W)
        batch_size, _, height, width = x.size()
        # Initialize h and c with torch.zeros
        if h is None:
            h = torch.zeros(size=(batch_size, self.filters, height, width)).type_as(x)
        if c is None:
            c = torch.zeros(size=(batch_size, self.filters, height, width)).type_as(x)
        # forward process
        i, f, g, o = torch.split(self.conv(torch.cat([x, h], dim=1)), self.filters, dim=1)
        i, f, g, o = torch.sigmoid(i), torch.sigmoid(f), torch.tanh(g), torch.sigmoid(o)
        c = f * c + i * g
        h = o * torch.tanh(c)
        
        return h, c


class Down(nn.Module):
    """Convolutional cell for 5D (S, B, C, H, W) input. The Down consists of 2 parts, 
        the ResNet bottleneck and the SENet module (optional). 
        
    Args:
        channels (int): Number of input channels.
        filters (int): Number of convolutional kernels.
        kernel_size (int, tuple): Size of convolutional kernels.
        stride (int, tuple): Stride of the convolution.
        padding (int, tuple): Padding of the convolution.
    """
    
    def __init__(self, channels: int, filters: int, kernel_size: Union[int, tuple] = 3, padding: Union[int, tuple] = 1):
        super(Down, self).__init__()

        self.down = nn.Sequential(
            nn.Conv2d(channels, filters, kernel_size=kernel_size, stride=2, padding=padding),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        return x


class Up(nn.Module):
    """Bilinear-resize-convolutional cell for 5D (S, B, C, H, W) input. 
        
    Args:
        channels (int): Number of input channels.
        filters (int): Number of convolutional kernels.
        kernel_size (int, tuple): Size of convolutional kernels.
        padding (int, tuple): Padding of the convolution.
    """
    
    def __init__(self, channels: int, filters: int, kernel_size: Union[int, tuple] = 3, 
                 padding: Union[int, tuple] = 1):
        super(Up, self).__init__()

        self.up = nn.Sequential(
            nn.ConvTranspose2d(channels, filters, kernel_size=kernel_size, stride=2, padding=padding, output_padding=1),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        return x
