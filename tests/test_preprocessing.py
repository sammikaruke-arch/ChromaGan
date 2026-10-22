"""
Unit Tests for Image Preprocessing and CIE Lab Conversions.
"""

import os
import sys
import numpy as np
import torch
import pytest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.image_utils import (
    rgb_to_lab_tensors,
    lab_tensors_to_rgb,
    prepare_input_for_inference,
    reconstruct_full_resolution,
)
from utils.metrics import compute_mae, compute_psnr, compute_ssim, evaluate_prediction


class TestPreprocessing:
    def test_rgb_to_lab_shapes_and_ranges(self):
        """Test RGB to Lab tensor conversion shapes and normalization bounds [-1, 1]."""
        # Create synthetic RGB float image [256, 256, 3]
        dummy_rgb = np.random.uniform(0.0, 1.0, (256, 256, 3)).astype(np.float32)
        l_tensor, ab_tensor = rgb_to_lab_tensors(dummy_rgb)

        assert l_tensor.shape == (1, 256, 256), f"Unexpected L shape: {l_tensor.shape}"
        assert ab_tensor.shape == (2, 256, 256), f"Unexpected ab shape: {ab_tensor.shape}"

        # CIE L is in [0, 100], normalized by (L / 50) - 1 -> should strictly lie in [-1.0, 1.0]
        assert -1.01 <= l_tensor.min().item() <= 1.01
        assert -1.01 <= l_tensor.max().item() <= 1.01

        # ab channels normalized by 128.0 should lie within approx [-1.0, 1.0]
        assert -1.5 <= ab_tensor.min().item() <= 1.5
        assert -1.5 <= ab_tensor.max().item() <= 1.5

    def test_lab_to_rgb_roundtrip(self):
        """Test that converting RGB -> Lab -> RGB preserves image structure with negligible error."""
        # Solid primary color patches
        test_rgb = np.zeros((64, 64, 3), dtype=np.float32)
        test_rgb[:32, :32] = [1.0, 0.0, 0.0]  # Red
        test_rgb[:32, 32:] = [0.0, 1.0, 0.0]  # Green
        test_rgb[32:, :32] = [0.0, 0.0, 1.0]  # Blue
        test_rgb[32:, 32:] = [1.0, 1.0, 0.0]  # Yellow

        l_tensor, ab_tensor = rgb_to_lab_tensors(test_rgb)
        recon_rgb = lab_tensors_to_rgb(l_tensor, ab_tensor)

        assert recon_rgb.shape == (64, 64, 3)
        assert recon_rgb.dtype == np.uint8

        # Ground truth uint8
        gt_uint8 = (test_rgb * 255.0).astype(np.uint8)

        # PSNR should be extremely high (> 40 dB)
        psnr_val = compute_psnr(gt_uint8, recon_rgb)
        assert psnr_val > 35.0, f"Roundtrip PSNR too low: {psnr_val} dB"

    def test_reconstruct_full_resolution(self):
        """Test bicubic upsampling of predicted ab channels to arbitrary resolution."""
        orig_h, orig_w = 480, 640
        orig_l = np.random.uniform(-1.0, 1.0, (orig_h, orig_w)).astype(np.float32)
        pred_ab_256 = torch.zeros((1, 2, 256, 256), dtype=torch.float32)

        reconstructed = reconstruct_full_resolution(
            orig_l_norm=orig_l,
            predicted_ab_tensor=pred_ab_256,
            original_size=(orig_w, orig_h),
        )

        assert reconstructed.shape == (orig_h, orig_w, 3)
        assert reconstructed.dtype == np.uint8

    def test_metrics_evaluation(self):
        """Test evaluation metrics functions."""
        img1 = np.full((128, 128, 3), 100, dtype=np.uint8)
        img2 = np.full((128, 128, 3), 110, dtype=np.uint8)

        metrics = evaluate_prediction(img1, img2)
        assert "psnr" in metrics
        assert "ssim" in metrics
        assert "mae_rgb" in metrics
        assert metrics["mae_rgb"] == 10.0
        assert 20.0 < metrics["psnr"] < 35.0
