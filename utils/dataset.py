"""
PyTorch Dataset and DataLoader Pipeline for Image Colorization GAN.

Loads RGB photographic images, applies data augmentation,
converts images to CIE Lab color space, extracts L and ab channels,
and normalizes values to [-1.0, 1.0].
"""

import os
import sys
import random
from typing import Tuple, List, Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps, ImageDraw
from skimage.color import rgb2lab

import config


# Supported image extensions
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_valid_image(filepath: str) -> bool:
    """Check if file has a valid image extension and exists."""
    ext = os.path.splitext(filepath)[1].lower()
    return ext in VALID_EXTENSIONS and os.path.isfile(filepath)


class ColorizationDataset(Dataset):
    """
    PyTorch Dataset for Image Colorization.

    Yields:
        L: [1, H, W] tensor in range [-1.0, 1.0] (Grayscale input)
        ab: [2, H, W] tensor in range [-1.0, 1.0] (Color ground truth)
    """
    def __init__(
        self,
        image_paths: List[str],
        img_size: int = config.IMG_SIZE,
        is_train: bool = True,
    ):
        self.image_paths = [p for p in image_paths if is_valid_image(p)]
        self.img_size = img_size
        self.is_train = is_train

        if len(self.image_paths) == 0:
            raise ValueError("No valid image files found in the provided list.")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path)
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
        except Exception as e:
            # Fallback to a solid neutral image if corrupted
            img = Image.new("RGB", (self.img_size, self.img_size), color=(128, 128, 128))

        # Resize to target size
        img = img.resize((self.img_size, self.img_size), Image.Resampling.BICUBIC)

        # Data augmentation (only for training)
        if self.is_train:
            # Random horizontal flip
            if random.random() > 0.5:
                img = ImageOps.mirror(img)

        # Convert to float numpy array [0.0, 1.0]
        rgb_array = np.array(img, dtype=np.float32) / 255.0

        # Convert to CIE Lab color space
        lab = rgb2lab(rgb_array)
        l_chan = lab[:, :, 0]    # [0.0, 100.0]
        ab_chan = lab[:, :, 1:]  # [-128.0, 127.0]

        # Normalize to [-1.0, 1.0]
        l_norm = (l_chan / config.L_NORM_DIVISOR) - config.L_NORM_SUBTRACT
        ab_norm = ab_chan / config.AB_NORM_DIVISOR

        # Convert to PyTorch tensors
        l_tensor = torch.from_numpy(l_norm).float().unsqueeze(0)        # [1, H, W]
        ab_tensor = torch.from_numpy(ab_norm).float().permute(2, 0, 1)  # [2, H, W]

        return l_tensor, ab_tensor


def get_image_files(directory: str) -> List[str]:
    """Recursively discover all valid image files in a directory."""
    images = []
    if not os.path.exists(directory):
        return images
    for root, _, files in os.walk(directory):
        for f in files:
            p = os.path.join(root, f)
            if is_valid_image(p):
                images.append(p)
    return sorted(images)


def create_dataloaders(
    data_dir: str = config.DATA_DIR,
    batch_size: int = config.BATCH_SIZE,
    img_size: int = config.IMG_SIZE,
    val_split: float = 0.15,
    seed: int = 42,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, int, int]:
    """
    Creates train and validation DataLoaders.
    Checks if 'data/train' and 'data/val' subdirectories exist.
    If not, splits images found in `data_dir` deterministically with no leakage.

    Returns:
        (train_loader, val_loader, num_train, num_val)
    """
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    train_files = get_image_files(train_dir)
    val_files = get_image_files(val_dir)

    if len(train_files) > 0 and len(val_files) > 0:
        # User has explicit train/val subdirectories
        train_paths = train_files
        val_paths = val_files
    else:
        # Fall back to splitting all files in data_dir
        all_files = get_image_files(data_dir)
        if len(all_files) == 0:
            raise FileNotFoundError(
                f"No images found in {data_dir}. Please place images in data/train and data/val, "
                f"or run with --smoke-test to generate synthetic sample images."
            )
        random.seed(seed)
        shuffled = all_files.copy()
        random.shuffle(shuffled)
        split_idx = max(1, int(len(shuffled) * (1.0 - val_split)))
        train_paths = shuffled[:split_idx]
        val_paths = shuffled[split_idx:]
        if len(val_paths) == 0:
            val_paths = train_paths[:min(2, len(train_paths))]

    train_dataset = ColorizationDataset(train_paths, img_size=img_size, is_train=True)
    val_dataset = ColorizationDataset(val_paths, img_size=img_size, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=len(train_dataset) > batch_size,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return train_loader, val_loader, len(train_dataset), len(val_dataset)


def generate_sample_dataset(output_dir: str = config.DATA_DIR, count: int = 16) -> None:
    """
    Generates realistic synthetic sample images (landscapes, gradients, geometric patterns)
    for smoke-testing the pipeline and for instant local demonstrations without external downloads.
    """
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    samples_dir = config.SAMPLES_DIR

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)

    colors = [
        ((34, 139, 34), (135, 206, 235)),   # Forest & Sky
        ((255, 69, 0), (255, 215, 0)),     # Sunset Orange & Gold
        ((70, 130, 180), (240, 230, 140)),  # Ocean Blue & Sand
        ((128, 0, 128), (255, 105, 180)),  # Violet & Pink
        ((220, 20, 60), (30, 144, 255)),   # Crimson & Dodger Blue
        ((0, 100, 0), (255, 165, 0)),      # Deep Green & Orange
    ]

    for i in range(count):
        # Create a 256x256 composite scene
        img = Image.new("RGB", (config.IMG_SIZE, config.IMG_SIZE))
        draw = ImageDraw.Draw(img)

        c1, c2 = colors[i % len(colors)]

        # Background vertical gradient
        for y in range(config.IMG_SIZE):
            ratio = y / config.IMG_SIZE
            r = int(c1[0] * (1 - ratio) + c2[0] * ratio)
            g = int(c1[1] * (1 - ratio) + c2[1] * ratio)
            b = int(c1[2] * (1 - ratio) + c2[2] * ratio)
            draw.line([(0, y), (config.IMG_SIZE, y)], fill=(r, g, b))

        # Add geometric foreground elements (sun, mountains, circles)
        cx, cy = (50 + (i * 37) % 150, 40 + (i * 23) % 90)
        draw.ellipse([cx - 25, cy - 25, cx + 25, cy + 25], fill=(255, 240, 180))

        # Polygon mountain / hill
        pts = [
            (0, config.IMG_SIZE),
            (60 + (i * 20) % 80, 140),
            (160 + (i * 15) % 60, 110),
            (config.IMG_SIZE, config.IMG_SIZE),
        ]
        draw.polygon(pts, fill=(int(c1[0] * 0.7), int(c1[1] * 0.7), int(c1[2] * 0.7)))

        # Split: 80% train, 20% val
        target_dir = val_dir if (i % 5 == 0) else train_dir
        filename = f"sample_{i:03d}.jpg"
        save_path = os.path.join(target_dir, filename)
        img.save(save_path, quality=95)

        # Also save to static/samples for web UI demo picker
        if i < 4:
            img.save(os.path.join(samples_dir, f"demo_{i+1}.jpg"), quality=95)


if __name__ == "__main__":
    print("Testing dataset module...")
    generate_sample_dataset(count=10)
    train_loader, val_loader, n_train, n_val = create_dataloaders(batch_size=4)
    print(f"DataLoaders created successfully: {n_train} train images, {n_val} val images.")
    for l_b, ab_b in train_loader:
        print(f"Batch L shape:  {l_b.shape}, range: [{l_b.min():.2f}, {l_b.max():.2f}]")
        print(f"Batch ab shape: {ab_b.shape}, range: [{ab_b.min():.2f}, {ab_b.max():.2f}]")
        break
    print("Dataset module sanity check PASSED.")
