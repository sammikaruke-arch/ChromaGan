"""
Configuration settings for the Black & White to Color Image GAN project.
Includes hyperparameters, file paths, model configurations, and web settings.
"""

import os
import torch

# Base Directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image & Color Space Settings
IMG_SIZE = 256                  # Spatial dimension (256x256)
IN_CHANNELS = 1                 # Input: L channel (grayscale)
OUT_CHANNELS = 2                # Output: ab channels (color components)

# Training Hyperparameters
BATCH_SIZE = 8 if DEVICE.type == "cuda" else 4
LEARNING_RATE_G = 2e-4          # Generator learning rate
LEARNING_RATE_D = 2e-4          # Discriminator learning rate
BETA1 = 0.5                     # Adam optimizer beta1
BETA2 = 0.999                   # Adam optimizer beta2
LAMBDA_L1 = 100.0               # L1 reconstruction loss weighting factor
NUM_EPOCHS = 40                 # Default training epochs
SAVE_FREQ = 5                   # Save checkpoint every N epochs
SAMPLE_FREQ = 1                 # Save sample generation images every N epochs

# Directory Paths
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOAD_DIR = os.path.join(STATIC_DIR, "uploads")
OUTPUT_DIR = os.path.join(STATIC_DIR, "outputs")
SAMPLES_DIR = os.path.join(STATIC_DIR, "samples")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

# Checkpoint Filenames
BEST_GENERATOR_PATH = os.path.join(CHECKPOINT_DIR, "best_generator.pth")
LATEST_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pth")
INITIAL_WEIGHTS_PATH = os.path.join(CHECKPOINT_DIR, "init_generator.pth")

# Web Application Settings
SECRET_KEY = os.environ.get("SECRET_KEY", "gan-colorization-secret-key-2026")
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}

# Color Space Normalization Constants
# CIE Lab color space ranges:
# L: [0, 100]  -> Normalized to [-1.0, 1.0] via (L / 50.0) - 1.0
# a: [-128, 127] -> Normalized to [-1.0, 1.0] via ab / 128.0
# b: [-128, 127] -> Normalized to [-1.0, 1.0] via ab / 128.0
L_NORM_DIVISOR = 50.0
L_NORM_SUBTRACT = 1.0
AB_NORM_DIVISOR = 128.0
