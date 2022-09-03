import torch
import torch.nn as nn
from torch.distributions import Normal
from .layers import Down, Up


class AttnUNet(nn.Module):
    def __init__(self, input_steps: int, forecast_steps: int, add_noise: bool = False):
        super(AttnUNet, self).__init__()
        self.input_steps = input_steps
        self.forecast_steps = forecast_steps
        self.add_noise = add_noise

        self.in_conv = nn.Conv2d(input_steps, 32, kernel_size=1)

        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 256)

        if self.add_noise:
            self.up4 = Up(768, 128)
        else:
            self.up4 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up2 = Up(128, 32)
        self.up1 = Up(64, 32)

        self.out_conv = nn.Conv2d(32, forecast_steps, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        length, batch_size, channels, height, width = x.size()
        h = x.transpose(1, 0).reshape(batch_size, length * channels, height, width)

        # Input step
        h1 = self.in_conv(h)
        h2 = self.down1(h1)
        h3 = self.down2(h2)
        h4 = self.down3(h3)
        h_last = self.down4(h4)
        
        # Forecast step
        if self.add_noise:
            z = Normal(0, 1).sample(h_last.size()).type_as(h_last)
            h_last = torch.cat([h_last, z], dim=1)
        h4p = self.up4(h_last, h4)
        h3p = self.up3(h4p, h3)
        h2p = self.up2(h3p, h2)
        h1p = self.up1(h2p, h1)

        out = self.out_conv(h1p)
        out = out.reshape(batch_size, -1, channels, height, width).transpose(1, 0)
        out = self.relu(out)
        return out
        