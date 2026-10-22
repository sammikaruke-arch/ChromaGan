"""
Image Preprocessing, Color Space Conversion, and Postprocessing Utilities.

Implements RGB <-> CIE Lab transformations, tensor normalization [-1, 1],
high-resolution luminance preservation, and side-by-side comparison helpers.
"""

import os
import sys
from typing import Tuple, Union, Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from skimage.color import rgb2lab, lab2rgb

import config


def load_and_validate_image(
    image_source: Union[str, bytes, Image.Image],
    target_size: Optional[Tuple[int, int]] = (config.IMG_SIZE, config.IMG_SIZE),
) -> Tuple[Image.Image, np.ndarray]:
    """
    Load an image from a file path, byte stream, or PIL Image.
    Handles RGB, RGBA, and Grayscale modes, corrects EXIF orientation,
    and returns both the normalized RGB image and the resized PIL image.

    Returns:
        (pil_image_rgb, rgb_numpy_float)
        rgb_numpy_float is in range [0.0, 1.0] with shape [H, W, 3]
    """
    if isinstance(image_source, Image.Image):
        img = image_source
    elif isinstance(image_source, str):
        if not os.path.exists(image_source):
            raise FileNotFoundError(f"Image file not found: {image_source}")
        img = Image.open(image_source)
    else:
        import io
        img = Image.open(io.BytesIO(image_source))

    # Correct orientation if EXIF tag exists
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass

    # Convert to RGB (handles RGBA, P, L, 1 modes cleanly)
    img = img.convert("RGB")

    if target_size is not None:
        img_resized = img.resize(target_size, Image.Resampling.BICUBIC)
    else:
        img_resized = img

    rgb_array = np.array(img_resized, dtype=np.float32) / 255.0
    return img_resized, rgb_array


def rgb_to_lab_tensors(
    rgb_array: np.ndarray,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert an RGB float array [0, 1] of shape [H, W, 3] or [B, H, W, 3]
    to normalized PyTorch tensors:
    - L channel: [1, H, W] in range [-1.0, 1.0]
    - ab channels: [2, H, W] in range [-1.0, 1.0]
    """
    if rgb_array.ndim == 3:
        # Single image: [H, W, 3]
        lab = rgb2lab(rgb_array)
        l_chan = lab[:, :, 0]             # [0, 100]
        ab_chan = lab[:, :, 1:]           # approx [-128, 127]

        # Normalize to [-1.0, 1.0]
        l_norm = (l_chan / config.L_NORM_DIVISOR) - config.L_NORM_SUBTRACT
        ab_norm = ab_chan / config.AB_NORM_DIVISOR

        l_tensor = torch.from_numpy(l_norm).float().unsqueeze(0)   # [1, H, W]
        ab_tensor = torch.from_numpy(ab_norm).float().permute(2, 0, 1)  # [2, H, W]
        return l_tensor, ab_tensor

    elif rgb_array.ndim == 4:
        # Batch: [B, H, W, 3]
        batch_l = []
        batch_ab = []
        for i in range(rgb_array.shape[0]):
            lab = rgb2lab(rgb_array[i])
            l_chan = lab[:, :, 0]
            ab_chan = lab[:, :, 1:]

            l_norm = (l_chan / config.L_NORM_DIVISOR) - config.L_NORM_SUBTRACT
            ab_norm = ab_chan / config.AB_NORM_DIVISOR

            batch_l.append(torch.from_numpy(l_norm).float().unsqueeze(0))
            batch_ab.append(torch.from_numpy(ab_norm).float().permute(2, 0, 1))

        return torch.stack(batch_l, dim=0), torch.stack(batch_ab, dim=0)
    else:
        raise ValueError(f"Expected 3D or 4D array, got shape: {rgb_array.shape}")


def lab_tensors_to_rgb(
    l_norm: Union[torch.Tensor, np.ndarray],
    ab_norm: Union[torch.Tensor, np.ndarray],
) -> np.ndarray:
    """
    Reconstruct RGB image [H, W, 3] (uint8 0-255) from normalized L and ab channels.

    Args:
        l_norm: [1, H, W] or [H, W] normalized in [-1.0, 1.0]
        ab_norm: [2, H, W] or [H, W, 2] normalized in [-1.0, 1.0]
    Returns:
        np.ndarray of shape [H, W, 3], uint8 range [0, 255]
    """
    # Convert PyTorch tensors to numpy if needed
    if isinstance(l_norm, torch.Tensor):
        l_norm = l_norm.detach().cpu().numpy()
    if isinstance(ab_norm, torch.Tensor):
        ab_norm = ab_norm.detach().cpu().numpy()

    # Squeeze batch or channel dimension from L if present: [1, H, W] -> [H, W]
    if l_norm.ndim == 3 and l_norm.shape[0] == 1:
        l_norm = l_norm.squeeze(0)

    # Convert ab from [2, H, W] to [H, W, 2] if needed
    if ab_norm.ndim == 3 and ab_norm.shape[0] == 2:
        ab_norm = np.transpose(ab_norm, (1, 2, 0))

    # Denormalize to CIE Lab ranges
    l_denorm = np.clip((l_norm + config.L_NORM_SUBTRACT) * config.L_NORM_DIVISOR, 0.0, 100.0)
    ab_denorm = np.clip(ab_norm * config.AB_NORM_DIVISOR, -128.0, 127.0)

    # Combine into Lab array [H, W, 3]
    lab = np.zeros((l_denorm.shape[0], l_denorm.shape[1], 3), dtype=np.float64)
    lab[:, :, 0] = l_denorm
    lab[:, :, 1:] = ab_denorm

    # Convert Lab to RGB [0.0, 1.0]
    rgb_float = lab2rgb(lab)
    rgb_uint8 = (np.clip(rgb_float, 0.0, 1.0) * 255.0).astype(np.uint8)
    return rgb_uint8


def prepare_input_for_inference(
    image_path: str,
    target_size: int = config.IMG_SIZE,
) -> Tuple[torch.Tensor, np.ndarray, Tuple[int, int]]:
    """
    Prepares any user-uploaded image for the generator.
    Preserves original high-resolution lightness channel for crisp post-reconstruction.

    Returns:
        model_input_tensor: [1, 1, target_size, target_size] normalized [-1, 1]
        original_l_norm: [orig_H, orig_W] normalized [-1, 1]
        original_size: (width, height)
    """
    img = Image.open(image_path)
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    img = img.convert("RGB")
    orig_w, orig_h = img.size

    # Original high-res Lab conversion
    orig_rgb = np.array(img, dtype=np.float32) / 255.0
    orig_lab = rgb2lab(orig_rgb)
    orig_l = orig_lab[:, :, 0]
    orig_l_norm = (orig_l / config.L_NORM_DIVISOR) - config.L_NORM_SUBTRACT

    # Model input: resized to target_size x target_size
    img_resized = img.resize((target_size, target_size), Image.Resampling.BICUBIC)
    resized_rgb = np.array(img_resized, dtype=np.float32) / 255.0
    resized_lab = rgb2lab(resized_rgb)
    resized_l = resized_lab[:, :, 0]
    resized_l_norm = (resized_l / config.L_NORM_DIVISOR) - config.L_NORM_SUBTRACT

    model_input = torch.from_numpy(resized_l_norm).float().unsqueeze(0).unsqueeze(0)  # [1, 1, 256, 256]
    return model_input, orig_l_norm, (orig_w, orig_h)


def reconstruct_full_resolution(
    orig_l_norm: np.ndarray,
    predicted_ab_tensor: torch.Tensor,
    original_size: Tuple[int, int],
) -> np.ndarray:
    """
    Upscales the generator's predicted 256x256 ab channels back to the original image
    resolution using bicubic interpolation, and merges them with the original high-resolution
    lightness (L) channel. This maintains full photographic sharpness!

    Args:
        orig_l_norm: [H_orig, W_orig]
        predicted_ab_tensor: [1, 2, 256, 256] in range [-1, 1]
        original_size: (width, height)
    Returns:
        RGB uint8 numpy array [H_orig, W_orig, 3]
    """
    orig_w, orig_h = original_size

    # Upsample predicted ab tensor to (orig_h, orig_w)
    ab_upsampled = F.interpolate(
        predicted_ab_tensor,
        size=(orig_h, orig_w),
        mode="bicubic",
        align_corners=True,
    ).squeeze(0)  # [2, orig_h, orig_w]

    # Convert to RGB with full-resolution L
    rgb_output = lab_tensors_to_rgb(orig_l_norm, ab_upsampled)
    return rgb_output


def create_comparison_grid(
    grayscale_rgb: np.ndarray,
    colorized_rgb: np.ndarray,
    ground_truth_rgb: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Creates a side-by-side comparison image:
    [Grayscale Input] | [Colorized GAN Output] (| [Ground Truth])
    """
    panels = [grayscale_rgb, colorized_rgb]
    if ground_truth_rgb is not None:
        panels.append(ground_truth_rgb)

    # Ensure all panels have the same height
    target_h = panels[0].shape[0]
    aligned_panels = []
    for p in panels:
        if p.shape[0] != target_h:
            h, w = p.shape[:2]
            new_w = int(w * (target_h / h))
            p_pil = Image.fromarray(p).resize((new_w, target_h), Image.Resampling.BICUBIC)
            aligned_panels.append(np.array(p_pil))
        else:
            aligned_panels.append(p)

    # Add a thin white separator line (3 pixels) between panels
    separator = np.full((target_h, 4, 3), 255, dtype=np.uint8)
    combined = []
    for i, p in enumerate(aligned_panels):
        combined.append(p)
        if i < len(aligned_panels) - 1:
            combined.append(separator)

    return np.concatenate(combined, axis=1)
