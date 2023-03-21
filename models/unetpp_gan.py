# refers to https://github.com/avgeiss/radar_inpainting
import torch
import torch.nn as nn


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


class Concatenate(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.cat(x, dim=self.dim)


class UNetppConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        return out


class UNetpp_Generator(nn.Module):
    def __init__(self, in_channels: int, base_channels: int):
        super().__init__()
        self.conv00 = UNetppConv2d(in_channels, base_channels)
        self.conv10 = UNetppConv2d(base_channels, base_channels)
        self.conv20 = UNetppConv2d(base_channels, base_channels)
        self.conv30 = UNetppConv2d(base_channels, base_channels)
        self.conv01 = UNetppConv2d(base_channels * 2, base_channels)
        self.conv11 = UNetppConv2d(base_channels * 3, base_channels)
        self.conv21 = UNetppConv2d(base_channels * 3, base_channels)
        self.conv02 = UNetppConv2d(base_channels * 3, base_channels)
        self.conv12 = UNetppConv2d(base_channels * 4, base_channels)
        self.conv03 = UNetppConv2d(base_channels * 4, base_channels)
        self.downsampling = nn.MaxPool2d(2, 2)
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
        self.out_conv = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)
        self.act = nn.Tanh()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y00 = self.conv00(x)
        y10 = self.conv10(self.downsampling(y00))
        y20 = self.conv20(self.downsampling(y10))
        y30 = self.conv30(self.downsampling(y20))
        y01 = self.conv01(torch.cat([y00, self.upsampling(y10)], dim=1))
        y11 = self.conv11(torch.cat([y10, self.upsampling(y20), self.downsampling(y01)], dim=1))
        y21 = self.conv21(torch.cat([y20, self.upsampling(y30), self.downsampling(y11)], dim=1))
        y02 = self.conv02(torch.cat([y00, y01, self.upsampling(y11)], dim=1))
        y12 = self.conv12(torch.cat([y10, y11, self.upsampling(y21), self.downsampling(y02)], dim=1))
        y03 = self.conv12(torch.cat([y00, y01, y02, self.upsampling(y12)], dim=1))
        out = self.act(self.out_conv(y03))
        return out


class UNetpp_Discriminator(nn.Module):
    def __init__(self, in_channels: int, base_channels: int, img_h: int, img_w: int):
        super().__init__()
        self.downsampling = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.1)
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1)
        self.flatten = Flatten()
        self.linear = nn.Linear(base_channels * 4 * img_h//8 * img_w//8, 1)
        self.act = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.downsampling(self.conv1(x)))
        x = self.dropout(self.downsampling(self.conv2(x)))
        x = self.dropout(self.downsampling(self.conv3(x)))
        x = self.act(self.linear(self.flatten(x)))
        return x


class UNetpp_GAN(nn.Module):
    def __init__(self, args):
        super().__init__()
        input_dim = len(args.elevation_id) * 2
        self.generator = UNetpp_Generator(input_dim, base_channels=32)
        self.discriminator = UNetpp_Discriminator(
            in_channels=2,
            base_channels=64,
            img_h=args.azimuth_range[1] - args.azimuth_range[0],
            img_w=args.radial_range[1] - args.radial_range[0]
        )
        self.args = args

    def forward(self, x):
        return self.generator(x)
