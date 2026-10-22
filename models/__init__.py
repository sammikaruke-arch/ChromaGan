"""Models package for Colorization GAN and Pretrained Colorizers."""

from .generator import UNetGenerator
from .discriminator import PatchDiscriminator
from .gan import ColorizationGAN, GANLoss, init_weights
from .base_color import BaseColor
from .eccv16 import ECCVGenerator, eccv16
from .siggraph17 import SIGGRAPHGenerator, siggraph17

__all__ = [
    "UNetGenerator",
    "PatchDiscriminator",
    "ColorizationGAN",
    "GANLoss",
    "init_weights",
    "BaseColor",
    "ECCVGenerator",
    "eccv16",
    "SIGGRAPHGenerator",
    "siggraph17",
]
