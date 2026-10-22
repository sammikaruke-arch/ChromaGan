"""
U-Net Generator Architecture for Grayscale to Color Image Translation.

Based on the Pix2Pix conditional GAN architecture (Isola et al., 2017).
Takes a 1-channel grayscale L (lightness) image and predicts 2-channel ab (chrominance) components.
Uses skip connections between encoder and decoder to retain high-frequency spatial details.
"""

import torch
import torch.nn as nn


class UNetDown(nn.Module):
    """
    Encoder block: Conv2d -> [BatchNorm2d] -> LeakyReLU(0.2)
    Downsamples the spatial resolution by a factor of 2.
    """
    def __init__(self, in_channels: int, out_channels: int, normalize: bool = True):
        super(UNetDown, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=not normalize)
        ]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class UNetUp(nn.Module):
    """
    Decoder block: ConvTranspose2d -> BatchNorm2d -> [Dropout] -> ReLU
    Upsamples the spatial resolution by a factor of 2 and concatenates skip connections.
    """
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        # Concatenate along channel dimension
        return torch.cat([x, skip], dim=1)


class UNetGenerator(nn.Module):
    """
    U-Net Generator with skip connections for 256x256 image colorization.

    Input:
        x: [B, 1, 256, 256] - Grayscale L channel normalized to [-1, 1]
    Output:
        out: [B, 2, 256, 256] - Predicted ab color channels normalized to [-1, 1]
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 2):
        super(UNetGenerator, self).__init__()

        # Encoder (Downsampling)
        # 256x256 -> 128x128
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        # 128x128 -> 64x64
        self.down2 = UNetDown(64, 128)
        # 64x64 -> 32x32
        self.down3 = UNetDown(128, 256)
        # 32x32 -> 16x16
        self.down4 = UNetDown(256, 512)
        # 16x16 -> 8x8
        self.down5 = UNetDown(512, 512)
        # 8x8 -> 4x4
        self.down6 = UNetDown(512, 512)
        # 4x4 -> 2x2
        self.down7 = UNetDown(512, 512)
        # 2x2 -> 1x1 (Bottleneck)
        self.down8 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

        # Decoder (Upsampling with skip connections)
        # 1x1 -> 2x2 (skip with down7: 512 + 512 = 1024)
        self.up1 = UNetUp(512, 512, dropout=0.5)
        # 2x2 -> 4x4 (skip with down6: 512 + 512 = 1024)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        # 4x4 -> 8x8 (skip with down5: 512 + 512 = 1024)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        # 8x8 -> 16x16 (skip with down4: 512 + 512 = 1024)
        self.up4 = UNetUp(1024, 512, dropout=0.0)
        # 16x16 -> 32x32 (skip with down3: 256 + 256 = 512)
        self.up5 = UNetUp(1024, 256, dropout=0.0)
        # 32x32 -> 64x64 (skip with down2: 128 + 128 = 256)
        self.up6 = UNetUp(512, 128, dropout=0.0)
        # 64x64 -> 128x128 (skip with down1: 64 + 64 = 128)
        self.up7 = UNetUp(256, 64, dropout=0.0)

        # Final output layer: 128x128 -> 256x256 -> 2 channels (ab)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Output range [-1.0, 1.0] matching normalized ab channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder forward pass
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        # Decoder forward pass with skip connections
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        out = self.final(u7)
        return out


if __name__ == "__main__":
    # Quick sanity check
    model = UNetGenerator(in_channels=1, out_channels=2)
    dummy_input = torch.randn(2, 1, 256, 256)
    dummy_output = model(dummy_input)
    print(f"Generator Input Shape:  {dummy_input.shape}")
    print(f"Generator Output Shape: {dummy_output.shape}")
    assert dummy_output.shape == (2, 2, 256, 256), "Output shape mismatch!"
    print("Generator architecture sanity check PASSED.")
