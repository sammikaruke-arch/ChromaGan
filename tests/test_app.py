"""
Integration and End-to-End Tests for Flask Web Application.
"""

import os
import io
import sys
from PIL import Image
import pytest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app
import config


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class TestFlaskEndpoints:
    def test_home_page_loads(self, client):
        """GET / should render the home page with status 200."""
        response = client.get("/")
        assert response.status_code == 200
        assert b"Chroma" in response.data
        assert b"Generative Adversarial Network" not in response.data

    def test_model_status_endpoint(self, client):
        """GET /status should return system information as JSON."""
        response = client.get("/status")
        assert response.status_code == 200
        data = response.get_json()
        assert "model_loaded" in data
        assert "device" in data

    def test_upload_no_file(self, client):
        """POST /colorize without a file should return 400 Bad Request."""
        response = client.post("/colorize", data={})
        assert response.status_code == 400
        data = response.get_json()
        assert data["success"] is False
        assert "error" in data

    def test_upload_invalid_extension(self, client):
        """POST /colorize with an invalid extension (e.g. .txt) should return 400."""
        dummy_txt = (io.BytesIO(b"dummy text content"), "test.txt")
        response = client.post("/colorize", data={"image": dummy_txt})
        assert response.status_code == 400
        data = response.get_json()
        assert "Unsupported file type" in data["error"]

    def test_colorize_valid_image(self, client):
        """POST /colorize with a valid image file should succeed if model is loaded."""
        # Create small test image
        img_byte_arr = io.BytesIO()
        img = Image.new("RGB", (128, 128), color=(150, 150, 150))
        img.save(img_byte_arr, format="JPEG")
        img_byte_arr.seek(0)

        response = client.post(
            "/colorize",
            data={"image": (img_byte_arr, "test_upload.jpg")},
            content_type="multipart/form-data",
        )

        data = response.get_json()
        if response.status_code == 200:
            assert data["success"] is True
            assert "output_url" in data
            assert "inference_time_ms" in data
        elif response.status_code == 503:
            # Model not trained / loaded in this specific environment
            assert "Model checkpoint not found" in data["error"]
        else:
            pytest.fail(f"Unexpected status code: {response.status_code}, data: {data}")

    def test_download_invalid_file(self, client):
        """GET /download/non_existent.png should return 404."""
        response = client.get("/download/definitely_not_existing_file.png")
        assert response.status_code == 404
