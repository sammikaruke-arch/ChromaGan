"""
Unit Tests for Colorizer Inference Engine.
"""

import os
import sys
import numpy as np
from PIL import Image
import pytest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from inference import Colorizer, ModelNotLoadedError


class TestInference:
    @pytest.fixture
    def test_image_path(self, tmp_path):
        """Creates a temporary sample image."""
        img_path = str(tmp_path / "test_input.jpg")
        img = Image.new("RGB", (300, 200), color=(120, 120, 120))
        img.save(img_path)
        return img_path

    def test_inference_with_trained_checkpoint(self, test_image_path, tmp_path):
        """Test complete inference pipeline using checkpoint."""
        colorizer = Colorizer()
        if not colorizer.is_loaded:
            pytest.skip("No trained checkpoint available yet.")

        out_path = str(tmp_path / "out_color.png")
        comp_path = str(tmp_path / "out_comp.png")

        res = colorizer.colorize(
            image_path=test_image_path,
            output_path=out_path,
            preserve_resolution=True,
            save_comparison=True,
            comparison_path=comp_path,
        )

        assert res["success"] is True
        assert os.path.exists(out_path)
        assert os.path.exists(comp_path)
        assert res["original_size"] == (300, 200)

        # Check output is valid 3-channel RGB image
        with Image.open(out_path) as out_img:
            assert out_img.mode == "RGB"
            assert out_img.size == (300, 200)

    def test_missing_checkpoint_handling(self, test_image_path):
        """Test error handling when pointing to non-existent checkpoint."""
        bad_colorizer = Colorizer(checkpoint_path="non_existent_weights.pth")
        assert bad_colorizer.is_loaded is False
        assert bad_colorizer.load_error_message is not None

        with pytest.raises(ModelNotLoadedError):
            bad_colorizer.colorize(test_image_path)

    def test_pretrained_eccv16_inference(self, test_image_path, tmp_path):
        """Test inference with pretrained ECCV16 model."""
        colorizer = Colorizer(default_model="eccv16")
        out_path = str(tmp_path / "eccv_out.png")
        res = colorizer.colorize(
            image_path=test_image_path,
            output_path=out_path,
            model_type="eccv16",
        )
        assert res["success"] is True
        assert os.path.exists(out_path)
        with Image.open(out_path) as out_img:
            assert out_img.mode == "RGB"
