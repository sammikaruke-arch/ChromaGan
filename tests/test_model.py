"""
Unit Tests for Generator, Discriminator, and GAN Loss Components.
"""

import os
import sys
import torch
import pytest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.generator import UNetGenerator
from models.discriminator import PatchDiscriminator
from models.gan import ColorizationGAN, GANLoss, init_weights
from models.eccv16 import eccv16
from models.siggraph17 import siggraph17


class TestModels:
    def test_generator_architecture_and_output_shape(self):
        """Test U-Net generator input and output tensor dimensions."""
        model = UNetGenerator(in_channels=1, out_channels=2)
        model.eval()

        batch_size = 2
        in_tensor = torch.randn(batch_size, 1, 256, 256)

        with torch.no_grad():
            out_tensor = model(in_tensor)

        assert out_tensor.shape == (batch_size, 2, 256, 256)
        # Tanh activation guarantees output in range [-1.0, 1.0]
        assert out_tensor.min() >= -1.0
        assert out_tensor.max() <= 1.0

    def test_discriminator_architecture_and_patch_shape(self):
        """Test PatchGAN discriminator conditioned on L and ab channels."""
        disc = PatchDiscriminator(in_channels=3)
        disc.eval()

        batch_size = 2
        l_tensor = torch.randn(batch_size, 1, 256, 256)
        ab_tensor = torch.randn(batch_size, 2, 256, 256)

        with torch.no_grad():
            patch_out = disc(l_tensor, ab_tensor)

        # 70x70 PatchGAN on 256x256 produces a 30x30 spatial grid
        assert patch_out.shape == (batch_size, 1, 30, 30)

    def test_gan_loss_lsgan_and_vanilla(self):
        """Test GAN loss functions for both real and fake targets."""
        # LSGAN Mode
        criterion_lsgan = GANLoss(gan_mode="lsgan")
        pred = torch.randn(2, 1, 30, 30)
        loss_real = criterion_lsgan(pred, target_is_real=True)
        loss_fake = criterion_lsgan(pred, target_is_real=False)

        assert loss_real.item() >= 0.0
        assert loss_fake.item() >= 0.0

        # Vanilla Mode (BCEWithLogits)
        criterion_vanilla = GANLoss(gan_mode="vanilla")
        loss_v_real = criterion_vanilla(pred, target_is_real=True)
        assert loss_v_real.item() >= 0.0

    def test_colorization_gan_training_step(self):
        """Test one full forward-backward optimization step of the ColorizationGAN system."""
        gan = ColorizationGAN(device=torch.device("cpu"))
        dummy_l = torch.randn(2, 1, 256, 256)
        dummy_ab = torch.randn(2, 2, 256, 256)

        gan.set_input(dummy_l, dummy_ab)
        losses = gan.optimize_parameters()

        assert "loss_d" in losses
        assert "loss_g" in losses
        assert "loss_g_gan" in losses
        assert "loss_g_l1" in losses

        assert losses["loss_d"] >= 0.0
        assert losses["loss_g"] >= 0.0

    def test_eccv16_architecture_and_output_shape(self):
        """Test ECCV16 generator forward pass and output dimensions."""
        model = eccv16(pretrained=False)
        model.eval()
        in_tensor = torch.randn(1, 1, 256, 256)
        with torch.no_grad():
            out = model(in_tensor)
        assert out.shape == (1, 2, 256, 256)

    def test_siggraph17_architecture_and_output_shape(self):
        """Test SIGGRAPH17 generator forward pass and output dimensions."""
        model = siggraph17(pretrained=False)
        model.eval()
        in_tensor = torch.randn(1, 1, 256, 256)
        with torch.no_grad():
            out = model(in_tensor)
        assert out.shape == (1, 2, 256, 256)
