"""
Inference and Image Colorization Module.

Supports:
1. Pretrained Deep Colorizers (ECCV16 & SIGGRAPH17) for immediate photorealistic colorization.
2. Custom Conditional GAN U-Net Generator (trained with Pix2Pix).
Maintains high-resolution luminance retention and batch evaluation.
"""

import os
import sys
import time
import argparse
from typing import Optional, Tuple, Dict, Any
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from skimage.color import rgb2lab, lab2rgb

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from models.generator import UNetGenerator
from models.eccv16 import eccv16, ECCVGenerator
from models.siggraph17 import siggraph17, SIGGRAPHGenerator
from utils.image_utils import (
    prepare_input_for_inference,
    reconstruct_full_resolution,
    lab_tensors_to_rgb,
    create_comparison_grid,
)


class ModelNotLoadedError(Exception):
    """Raised when inference is attempted without an available trained checkpoint."""
    pass


class Colorizer:
    """
    Colorization inference engine supporting both pretrained state-of-the-art
    deep colorizers (ECCV16 / SIGGRAPH17) and custom trained GAN U-Net generators.
    """
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        default_model: str = "eccv16",
    ):
        self.device = device or config.DEVICE
        self.default_model = default_model.lower()
        self.generator: Optional[UNetGenerator] = None
        self._eccv_model: Optional[ECCVGenerator] = None
        self._siggraph_model: Optional[SIGGRAPHGenerator] = None

        self.is_loaded = False
        self.checkpoint_path: Optional[str] = None
        self.load_error_message: Optional[str] = None

        # If a checkpoint path was explicitly provided, default to U-Net model
        if checkpoint_path is not None:
            self.default_model = "unet"
            self.load_checkpoint(checkpoint_path)
        else:
            # Check custom checkpoint availability
            custom_ckpt = self.find_default_checkpoint()
            if custom_ckpt:
                self.load_checkpoint(custom_ckpt)
            else:
                self.is_loaded = True  # Pretrained models available

    def find_default_checkpoint(self) -> Optional[str]:
        """Search default locations for a trained model checkpoint."""
        candidates = [
            config.BEST_GENERATOR_PATH,
            os.path.join(config.CHECKPOINT_DIR, "best_model.pth"),
            config.LATEST_CHECKPOINT_PATH,
        ]
        for candidate in candidates:
            if os.path.exists(candidate) and os.path.getsize(candidate) > 1000:
                return candidate
        return None

    def get_eccv_model(self) -> ECCVGenerator:
        """Lazy loader for pretrained ECCV16 generator."""
        if self._eccv_model is None:
            self._eccv_model = eccv16(pretrained=True).to(self.device).eval()
        return self._eccv_model

    def get_siggraph_model(self) -> SIGGRAPHGenerator:
        """Lazy loader for pretrained SIGGRAPH17 generator."""
        if self._siggraph_model is None:
            self._siggraph_model = siggraph17(pretrained=True).to(self.device).eval()
        return self._siggraph_model

    def load_checkpoint(self, checkpoint_path: Optional[str] = None) -> bool:
        """
        Loads custom generator weights from checkpoint.
        Handles both pure generator state dictionaries and full training checkpoint dicts.
        """
        target_path = checkpoint_path or self.find_default_checkpoint()

        if not target_path or not os.path.exists(target_path):
            self.is_loaded = False
            self.load_error_message = (
                f"No trained generator checkpoint found in '{config.CHECKPOINT_DIR}'. "
                "Please run training first (e.g. `python train.py --smoke_test` or full training) "
                "or place 'best_generator.pth' into the checkpoints directory."
            )
            return False

        try:
            if self.generator is None:
                self.generator = UNetGenerator(in_channels=1, out_channels=2).to(self.device)

            state = torch.load(target_path, map_location=self.device)
            if isinstance(state, dict) and "generator_state_dict" in state:
                self.generator.load_state_dict(state["generator_state_dict"])
            elif isinstance(state, dict):
                self.generator.load_state_dict(state)
            else:
                raise ValueError("Unrecognized checkpoint format.")

            self.generator.eval()
            self.is_loaded = True
            self.checkpoint_path = target_path
            self.load_error_message = None
            return True
        except Exception as e:
            self.is_loaded = False
            self.load_error_message = f"Failed to load checkpoint from {target_path}: {str(e)}"
            return False

    def _colorize_with_pretrained(
        self,
        image_path: str,
        model_name: str = "eccv16",
        preserve_resolution: bool = True,
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Run inference using pretrained ECCV16 or SIGGRAPH17 model."""
        model = self.get_eccv_model() if model_name == "eccv16" else self.get_siggraph_model()

        img = Image.open(image_path)
        try:
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass
        img = img.convert("RGB")
        orig_w, orig_h = img.size

        # High-res lightness extraction [0, 100]
        rgb_orig = np.array(img, dtype=np.float32) / 255.0
        lab_orig = rgb2lab(rgb_orig)
        l_orig = lab_orig[:, :, 0]

        # 256x256 model input
        img_rs = img.resize((config.IMG_SIZE, config.IMG_SIZE), Image.Resampling.BICUBIC)
        rgb_rs = np.array(img_rs, dtype=np.float32) / 255.0
        lab_rs = rgb2lab(rgb_rs)
        l_rs = lab_rs[:, :, 0]

        l_tensor = torch.from_numpy(l_rs).float().unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            predicted_ab = model(l_tensor)  # [1, 2, 256, 256]

        if preserve_resolution and (orig_w != config.IMG_SIZE or orig_h != config.IMG_SIZE):
            ab_high = F.interpolate(
                predicted_ab,
                size=(orig_h, orig_w),
                mode="bicubic",
                align_corners=True,
            ).squeeze(0).permute(1, 2, 0).cpu().numpy()

            lab_high = np.zeros((orig_h, orig_w, 3), dtype=np.float64)
            lab_high[:, :, 0] = l_orig
            lab_high[:, :, 1:] = ab_high
            colorized_rgb = (np.clip(lab2rgb(lab_high), 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            ab_np = predicted_ab.squeeze(0).permute(1, 2, 0).cpu().numpy()
            lab_np = np.zeros((config.IMG_SIZE, config.IMG_SIZE, 3), dtype=np.float64)
            lab_np[:, :, 0] = l_rs
            lab_np[:, :, 1:] = ab_np
            colorized_rgb = (np.clip(lab2rgb(lab_np), 0.0, 1.0) * 255.0).astype(np.uint8)

        return colorized_rgb, (orig_w, orig_h)

    def _colorize_with_unet(
        self,
        image_path: str,
        preserve_resolution: bool = True,
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Run inference using custom trained U-Net generator."""
        if not self.is_loaded or self.generator is None:
            raise ModelNotLoadedError(
                self.load_error_message or "Trained custom model checkpoint is not loaded."
            )

        model_input, orig_l_norm, (orig_w, orig_h) = prepare_input_for_inference(
            image_path, target_size=config.IMG_SIZE
        )
        model_input = model_input.to(self.device)

        self.generator.eval()
        with torch.no_grad():
            predicted_ab = self.generator(model_input)

        if preserve_resolution and (orig_w != config.IMG_SIZE or orig_h != config.IMG_SIZE):
            colorized_rgb = reconstruct_full_resolution(
                orig_l_norm=orig_l_norm,
                predicted_ab_tensor=predicted_ab,
                original_size=(orig_w, orig_h),
            )
        else:
            l_resized = model_input.squeeze(0).cpu()
            ab_out = predicted_ab.squeeze(0).cpu()
            colorized_rgb = lab_tensors_to_rgb(l_resized, ab_out)

        return colorized_rgb, (orig_w, orig_h)

    def colorize(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        preserve_resolution: bool = True,
        save_comparison: bool = False,
        comparison_path: Optional[str] = None,
        model_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Convert a grayscale image to a plausible color image.

        Args:
            image_path: Path to user-uploaded image.
            output_path: Optional destination file path.
            preserve_resolution: If True, merges upsampled ab channels with original high-res L channel.
            save_comparison: If True, also generates a side-by-side comparison image.
            comparison_path: Optional destination for comparison grid.
            model_type: "eccv16" (default), "siggraph17", or "unet" (custom GAN).
        """
        active_model = (model_type or self.default_model).lower()

        start_time = time.perf_counter()

        if active_model in ("eccv16", "siggraph17"):
            colorized_rgb, (orig_w, orig_h) = self._colorize_with_pretrained(
                image_path=image_path,
                model_name=active_model,
                preserve_resolution=preserve_resolution,
            )
        elif active_model in ("unet", "custom"):
            colorized_rgb, (orig_w, orig_h) = self._colorize_with_unet(
                image_path=image_path,
                preserve_resolution=preserve_resolution,
            )
        else:
            raise ValueError(f"Unknown model_type '{active_model}'. Choose 'eccv16', 'siggraph17', or 'unet'.")

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        colorized_pil = Image.fromarray(colorized_rgb)

        # Save colorized output
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            colorized_pil.save(output_path, quality=95)

        # Optional comparison side-by-side image
        saved_comp_path = None
        if save_comparison and comparison_path:
            gray_pil = Image.open(image_path).convert("L").convert("RGB")
            if preserve_resolution:
                gray_np = np.array(gray_pil)
            else:
                gray_np = np.array(gray_pil.resize((config.IMG_SIZE, config.IMG_SIZE), Image.Resampling.BICUBIC))

            comp_grid = create_comparison_grid(gray_np, colorized_rgb)
            os.makedirs(os.path.dirname(os.path.abspath(comparison_path)), exist_ok=True)
            Image.fromarray(comp_grid).save(comparison_path, quality=95)
            saved_comp_path = comparison_path

        return {
            "success": True,
            "colorized_image": colorized_pil,
            "output_path": output_path,
            "comparison_path": saved_comp_path,
            "original_size": (orig_w, orig_h),
            "inference_time_ms": round(elapsed_ms, 2),
            "device": self.device.type.upper(),
            "model_type": active_model,
        }


def main():
    parser = argparse.ArgumentParser(description="Colorize a Black & White Image")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input grayscale image")
    parser.add_argument("--output", "-o", type=str, default=None, help="Path to save colorized result")
    parser.add_argument("--model", "-m", type=str, default="eccv16", choices=["eccv16", "siggraph17", "unet"], help="Colorization model engine")
    parser.add_argument("--checkpoint", "-c", type=str, default=None, help="Path to custom generator checkpoint (for unet)")
    parser.add_argument("--no_preserve_res", action="store_true", help="Output fixed 256x256 image")
    args = parser.parse_args()

    colorizer = Colorizer(checkpoint_path=args.checkpoint, default_model=args.model)

    out_file = args.output
    if not out_file:
        base, ext = os.path.splitext(args.input)
        out_file = f"{base}_colorized.png"

    print(f"Colorizing image '{args.input}' using [{args.model.upper()}] model ...")
    result = colorizer.colorize(
        image_path=args.input,
        output_path=out_file,
        preserve_resolution=not args.no_preserve_res,
        save_comparison=True,
        comparison_path=f"{os.path.splitext(out_file)[0]}_comparison.png",
        model_type=args.model,
    )
    print("Colorization completed successfully!")
    print(f"Model Engine:        {result['model_type'].upper()}")
    print(f"Original Resolution: {result['original_size'][0]}x{result['original_size'][1]}")
    print(f"Inference Time:      {result['inference_time_ms']} ms ({result['device']})")
    print(f"Saved Output:        {result['output_path']}")
    if result["comparison_path"]:
        print(f"Saved Comparison:    {result['comparison_path']}")


if __name__ == "__main__":
    main()
