"""
PatchGAN Discriminator Architecture for Conditional GAN Image Colorization.

Based on the Pix2Pix conditional GAN discriminator (Isola et al., 2017).
Instead of penalizing the entire image with a single scalar, PatchGAN operates
on local 70x70 patches of the image to penalize structure at the scale of patches.
This enforces high-frequency local realism and sharp color boundaries.
"""

import torch
import torch.nn as nn


class PatchDiscriminator(nn.Module):
    """
    70x70 PatchGAN Discriminator.

    Conditions on:
        L: [B, 1, 256, 256] - Grayscale lightness channel
        ab: [B, 2, 256, 256] - Real or generated color channels
    Combined Input:
        Concatenated [L, ab] -> [B, 3, 256, 256]
    Output:
        Patch predictions -> [B, 1, 30, 30] (each element classifies a 70x70 receptive field)
    """
    def __init__(self, in_channels: int = 3, ndf: int = 64):
        """
        Args:
            in_channels: 1 (L channel) + 2 (ab channels) = 3 channels
            ndf: Number of discriminator filters in the first layer (default 64)
        """
        super(PatchDiscriminator, self).__init__()

        self.model = nn.Sequential(
            # Layer 1: 256x256 -> 128x128 (No normalization on first layer)
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 2: 128x128 -> 64x64
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 3: 64x64 -> 32x32
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 4: 32x32 -> 31x31 (stride=1 preserves receptive field)
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 5: 31x31 -> 30x30 output patch map
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1)
            # Note: No Sigmoid here; we use BCEWithLogitsLoss or MSELoss (LSGAN)
            # for improved numerical stability and training dynamics.
        )

    def forward(self, l_channel: torch.Tensor, ab_channels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            l_channel: Grayscale lightness channel [B, 1, H, W]
            ab_channels: Real or generated color channels [B, 2, H, W]
        Returns:
            Logits patch map [B, 1, H_patch, W_patch]
        """
        # Concatenate along channel dimension: [B, 1+2, H, W] = [B, 3, H, W]
        x = torch.cat([l_channel, ab_channels], dim=1)
        return self.model(x)


if __name__ == "__main__":
    # Quick sanity check
    discriminator = PatchDiscriminator(in_channels=3)
    dummy_l = torch.randn(2, 1, 256, 256)
    dummy_ab = torch.randn(2, 2, 256, 256)
    pred = discriminator(dummy_l, dummy_ab)
    print(f"Discriminator Input L Shape:  {dummy_l.shape}")
    print(f"Discriminator Input ab Shape: {dummy_ab.shape}")
    print(f"Discriminator Output Shape:   {pred.shape}")
    assert pred.shape == (2, 1, 30, 30), "Discriminator output shape mismatch!"
    print("PatchGAN Discriminator sanity check PASSED.")
