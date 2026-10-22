"""
Evaluation Metrics for Image Colorization.

Implements standard quantitative image quality assessment metrics:
- Mean Absolute Error (MAE) on ab chrominance channels
- Peak Signal-to-Noise Ratio (PSNR) in decibels (dB)
- Structural Similarity Index (SSIM)
- Delta E (CIE76 color difference in Lab space)

Note: All evaluation metrics require ground-truth original color photographs.
During real-world user inference on historic B&W photos, ground-truth color
does not exist, so colorization plausibility is assessed qualitatively.
"""

import os
import sys
from typing import Dict, Tuple, Optional
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compute_mae(pred_ab: np.ndarray, target_ab: np.ndarray) -> float:
    """
    Computes Mean Absolute Error between predicted and target ab channels.
    Both arrays are expected in normalized [-1.0, 1.0] or Lab scale.
    """
    return float(np.mean(np.abs(pred_ab - target_ab)))


def compute_psnr(image_true: np.ndarray, image_test: np.ndarray, data_range: int = 255) -> float:
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).
    Higher is better. Typically 20-35 dB for reconstructed images.
    """
    if np.array_equal(image_true, image_test):
        return 100.0  # Perfect match sentinel
    return float(peak_signal_noise_ratio(image_true, image_test, data_range=data_range))


def compute_ssim(image_true: np.ndarray, image_test: np.ndarray) -> float:
    """
    Computes Structural Similarity Index Measure (SSIM) in range [-1.0, 1.0].
    Higher is better (1.0 = identical structures).
    """
    return float(
        structural_similarity(
            image_true,
            image_test,
            channel_axis=2 if image_true.ndim == 3 else None,
            data_range=255,
        )
    )


def compute_delta_e(lab_true: np.ndarray, lab_pred: np.ndarray) -> float:
    """
    Computes CIE76 color difference Delta E:
    Delta E = sqrt((L1 - L2)^2 + (a1 - a2)^2 + (b1 - b2)^2)
    Lower is better (values < 2.3 are considered imperceptible to human eyes).
    """
    diff = lab_true.astype(np.float64) - lab_pred.astype(np.float64)
    delta_e = np.sqrt(np.sum(diff ** 2, axis=-1))
    return float(np.mean(delta_e))


def evaluate_prediction(
    ground_truth_rgb: np.ndarray,
    predicted_rgb: np.ndarray,
) -> Dict[str, float]:
    """
    Computes full evaluation suite between ground truth RGB and predicted RGB image.

    Args:
        ground_truth_rgb: [H, W, 3] uint8 (0-255)
        predicted_rgb: [H, W, 3] uint8 (0-255)
    Returns:
        Dict with keys: 'psnr', 'ssim', 'mae_rgb'
    """
    psnr_val = compute_psnr(ground_truth_rgb, predicted_rgb)
    ssim_val = compute_ssim(ground_truth_rgb, predicted_rgb)
    mae_val = float(np.mean(np.abs(ground_truth_rgb.astype(np.float32) - predicted_rgb.astype(np.float32))))

    return {
        "psnr": round(psnr_val, 2),
        "ssim": round(ssim_val, 4),
        "mae_rgb": round(mae_val, 2),
    }


if __name__ == "__main__":
    # Test evaluation module
    img1 = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    img2 = np.clip(img1.astype(np.int16) + np.random.randint(-10, 10, (256, 256, 3)), 0, 255).astype(np.uint8)

    metrics = evaluate_prediction(img1, img2)
    print("Metrics on perturbed image:", metrics)
    assert "psnr" in metrics and "ssim" in metrics
    print("Metrics module sanity check PASSED.")
