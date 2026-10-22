"""
Conditional GAN Model System for Image Colorization.

Encapsulates Generator, Discriminator, Loss functions, Optimizers,
and weight initialization. Implements the training step for cGAN.
"""

import os
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from .generator import UNetGenerator
from .discriminator import PatchDiscriminator


def init_weights(net: nn.Module, init_type: str = "normal", init_gain: float = 0.02) -> None:
    """
    Initialize network weights following the Pix2Pix paper specification:
    Normal distribution with mean 0.0 and std 0.02 for Conv layers,
    and mean 1.0, std 0.02 for BatchNorm layers.
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
            if init_type == "normal":
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif hasattr(m, "weight") and classname.find("BatchNorm2d") != -1:
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


class GANLoss(nn.Module):
    """
    Adversarial GAN loss supporting both LSGAN (MSELoss) and Vanilla GAN (BCEWithLogitsLoss).
    LSGAN (Least Squares GAN) is the default because it stabilizes training and
    penalizes samples based on their distance to the decision boundary.
    """
    def __init__(self, gan_mode: str = "lsgan", real_label: float = 1.0, fake_label: float = 0.0):
        super(GANLoss, self).__init__()
        self.register_buffer("real_label", torch.tensor(real_label))
        self.register_buffer("fake_label", torch.tensor(fake_label))
        self.gan_mode = gan_mode

        if gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        elif gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError(f"GAN mode '{gan_mode}' is not implemented.")

    def get_target_tensor(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        """Create target tensor filled with real or fake labels matching prediction shape."""
        target_val = self.real_label if target_is_real else self.fake_label
        return target_val.expand_as(prediction)

    def forward(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        return self.loss(prediction, target_tensor)


class ColorizationGAN:
    """
    Conditional GAN system for Image Colorization.

    Combines:
    - Generator: U-Net (L -> ab)
    - Discriminator: PatchGAN (L + ab -> patch grid)
    - Loss: GAN Adversarial Loss + Lambda * L1 Reconstruction Loss
    """
    def __init__(
        self,
        lr_g: float = 2e-4,
        lr_d: float = 2e-4,
        beta1: float = 0.5,
        beta2: float = 0.999,
        lambda_l1: float = 100.0,
        gan_mode: str = "lsgan",
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lambda_l1 = lambda_l1

        # Instantiate networks
        self.generator = UNetGenerator(in_channels=1, out_channels=2).to(self.device)
        self.discriminator = PatchDiscriminator(in_channels=3).to(self.device)

        # Initialize network weights
        init_weights(self.generator)
        init_weights(self.discriminator)

        # Loss functions
        self.criterion_gan = GANLoss(gan_mode=gan_mode).to(self.device)
        self.criterion_l1 = nn.L1Loss().to(self.device)

        # Optimizers
        self.optimizer_g = Adam(self.generator.parameters(), lr=lr_g, betas=(beta1, beta2))
        self.optimizer_d = Adam(self.discriminator.parameters(), lr=lr_d, betas=(beta1, beta2))

        # Internal state tensors
        self.real_l: Optional[torch.Tensor] = None
        self.real_ab: Optional[torch.Tensor] = None
        self.fake_ab: Optional[torch.Tensor] = None

        # Tracking losses
        self.loss_d: float = 0.0
        self.loss_g: float = 0.0
        self.loss_g_gan: float = 0.0
        self.loss_g_l1: float = 0.0

    def set_input(self, l_tensor: torch.Tensor, ab_tensor: torch.Tensor) -> None:
        """Transfer batch inputs to device."""
        self.real_l = l_tensor.to(self.device)
        self.real_ab = ab_tensor.to(self.device)

    def forward(self) -> torch.Tensor:
        """Run forward pass through the generator."""
        self.fake_ab = self.generator(self.real_l)
        return self.fake_ab

    def backward_d(self) -> torch.Tensor:
        """
        Calculate GAN loss for the Discriminator.
        D_loss = 0.5 * (loss_real + loss_fake)
        """
        # Real loss: Conditioned on real_l and real_ab
        pred_real = self.discriminator(self.real_l, self.real_ab)
        loss_d_real = self.criterion_gan(pred_real, target_is_real=True)

        # Fake loss: Conditioned on real_l and fake_ab (detached to avoid G gradients)
        pred_fake = self.discriminator(self.real_l, self.fake_ab.detach())
        loss_d_fake = self.criterion_gan(pred_fake, target_is_real=False)

        loss_d = (loss_d_real + loss_d_fake) * 0.5
        loss_d.backward()
        self.loss_d = loss_d.item()
        return loss_d

    def backward_g(self) -> torch.Tensor:
        """
        Calculate loss for the Generator:
        G_loss = loss_G_GAN + lambda_L1 * loss_G_L1
        """
        # Generator wants Discriminator to believe fake_ab is real
        pred_fake = self.discriminator(self.real_l, self.fake_ab)
        loss_g_gan = self.criterion_gan(pred_fake, target_is_real=True)

        # L1 color reconstruction loss between fake ab and real ab
        loss_g_l1 = self.criterion_l1(self.fake_ab, self.real_ab) * self.lambda_l1

        loss_g = loss_g_gan + loss_g_l1
        loss_g.backward()

        self.loss_g = loss_g.item()
        self.loss_g_gan = loss_g_gan.item()
        self.loss_g_l1 = loss_g_l1.item()
        return loss_g

    def optimize_parameters(self) -> Dict[str, float]:
        """
        One complete optimization step:
        1. Forward pass
        2. Discriminator update
        3. Generator update
        """
        self.forward()

        # Update Discriminator
        self.optimizer_d.zero_grad()
        self.backward_d()
        self.optimizer_d.step()

        # Update Generator
        self.optimizer_g.zero_grad()
        self.backward_g()
        self.optimizer_g.step()

        return {
            "loss_d": self.loss_d,
            "loss_g": self.loss_g,
            "loss_g_gan": self.loss_g_gan,
            "loss_g_l1": self.loss_g_l1,
        }

    def save_checkpoint(self, filepath: str, epoch: int, is_best: bool = False) -> None:
        """Save complete training state."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "generator_state_dict": self.generator.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
            "optimizer_g_state_dict": self.optimizer_g.state_dict(),
            "optimizer_d_state_dict": self.optimizer_d.state_dict(),
            "is_best": is_best,
        }
        torch.save(checkpoint, filepath)

    def save_generator_only(self, filepath: str) -> None:
        """Save only the generator weights for lightweight deployment & inference."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.generator.state_dict(), filepath)

    def load_checkpoint(self, filepath: str, resume_training: bool = True) -> int:
        """
        Load checkpoint.
        Returns the saved epoch number.
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        if isinstance(checkpoint, dict) and "generator_state_dict" in checkpoint:
            self.generator.load_state_dict(checkpoint["generator_state_dict"])
            if resume_training and "discriminator_state_dict" in checkpoint:
                self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
                self.optimizer_g.load_state_dict(checkpoint["optimizer_g_state_dict"])
                self.optimizer_d.load_state_dict(checkpoint["optimizer_d_state_dict"])
            return checkpoint.get("epoch", 0)
        else:
            # File is just the generator state dict
            self.generator.load_state_dict(checkpoint)
            return 0
