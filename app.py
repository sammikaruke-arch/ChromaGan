"""
Flask Web Application for Black & White to Color Image Conversion Using Deep Learning & GANs.

Provides a responsive web interface for:
- Drag-and-drop image upload
- Live interactive before/after comparison slider
- Sample image gallery for one-click testing
- Real-time deep learning colorization inference (ECCV16 / SIGGRAPH17 / Custom GAN)
- High-resolution colorized image downloads
- Architecture and educational explanations
"""

import os
import sys
import uuid
import shutil
from typing import Dict, Any
from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
from werkzeug.utils import secure_filename
from PIL import Image

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from inference import Colorizer, ModelNotLoadedError
from utils.dataset import generate_sample_dataset

app = Flask(__name__)
app.config["SECRET_KEY"] = config.SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH
app.config["UPLOAD_FOLDER"] = config.UPLOAD_DIR
app.config["OUTPUT_FOLDER"] = config.OUTPUT_DIR

# Ensure required directories exist
os.makedirs(config.UPLOAD_DIR, exist_ok=True)
os.makedirs(config.OUTPUT_DIR, exist_ok=True)
os.makedirs(config.SAMPLES_DIR, exist_ok=True)

# Initialize Colorization Engine (defaults to pretrained ECCV16 for immediate rich colors)
colorizer = Colorizer(default_model="eccv16")

# Check and generate sample images if missing
sample_files = [f for f in os.listdir(config.SAMPLES_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
if len(sample_files) == 0:
    generate_sample_dataset(output_dir=config.DATA_DIR, count=8)


def is_allowed_file(filename: str) -> bool:
    """Validate file extension against whitelist."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in config.ALLOWED_EXTENSIONS


@app.route("/")
def index():
    """Render main web application interface."""
    # Discover available demo sample images
    samples = []
    if os.path.exists(config.SAMPLES_DIR):
        for f in sorted(os.listdir(config.SAMPLES_DIR)):
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                samples.append({
                    "name": f,
                    "url": url_for("static", filename=f"samples/{f}"),
                })

    return render_template(
        "index.html",
        model_loaded=colorizer.is_loaded,
        checkpoint_path=colorizer.checkpoint_path or "Pretrained Deep Model",
        device=colorizer.device.type.upper(),
        samples=samples,
        default_model=colorizer.default_model,
    )


@app.route("/status", methods=["GET"])
def model_status():
    """API endpoint to query system and model health."""
    return jsonify({
        "model_loaded": colorizer.is_loaded,
        "device": colorizer.device.type.upper(),
        "checkpoint": colorizer.checkpoint_path,
        "load_error": colorizer.load_error_message,
        "default_model": colorizer.default_model,
        "available_models": ["eccv16", "siggraph17", "unet"],
    })


@app.route("/colorize", methods=["POST"])
def colorize_image():
    """
    Handle user image upload and run colorization inference.
    Supports selectable engine: "eccv16" (default), "siggraph17", or "unet" (custom GAN).
    """
    model_type = request.form.get("model_type", "eccv16").lower()

    # Check if custom model is requested but weights are missing
    if model_type in ("unet", "custom") and not colorizer.is_loaded:
        if not colorizer.load_checkpoint():
            return jsonify({
                "success": False,
                "error": "Custom GAN checkpoint not found. Please train the model using `python train.py` first, or choose the Pretrained ECCV16 model.",
                "details": colorizer.load_error_message,
            }), 503

    # Check file in request
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image file provided in upload request."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"success": False, "error": "Empty filename received."}), 400

    if not is_allowed_file(file.filename):
        allowed_str = ", ".join(sorted(config.ALLOWED_EXTENSIONS)).upper()
        return jsonify({
            "success": False,
            "error": f"Unsupported file type. Please upload an image with extension: {allowed_str}",
        }), 400

    try:
        # Secure unique filename generation
        file_ext = secure_filename(file.filename).rsplit(".", 1)[1].lower()
        unique_id = uuid.uuid4().hex
        input_filename = f"{unique_id}_input.{file_ext}"
        output_filename = f"{unique_id}_colorized.png"
        comparison_filename = f"{unique_id}_comparison.png"

        input_path = os.path.join(config.UPLOAD_DIR, input_filename)
        output_path = os.path.join(config.OUTPUT_DIR, output_filename)
        comparison_path = os.path.join(config.OUTPUT_DIR, comparison_filename)

        # Save uploaded file
        file.save(input_path)

        # Validate that image can be opened by PIL
        try:
            with Image.open(input_path) as img_check:
                img_check.verify()
        except Exception:
            if os.path.exists(input_path):
                os.remove(input_path)
            return jsonify({"success": False, "error": "Uploaded file is corrupted or not a valid image."}), 400

        # Execute colorization
        result = colorizer.colorize(
            image_path=input_path,
            output_path=output_path,
            preserve_resolution=True,
            save_comparison=True,
            comparison_path=comparison_path,
            model_type=model_type,
        )

        return jsonify({
            "success": True,
            "input_url": url_for("static", filename=f"uploads/{input_filename}"),
            "output_url": url_for("static", filename=f"outputs/{output_filename}"),
            "comparison_url": url_for("static", filename=f"outputs/{comparison_filename}"),
            "download_filename": output_filename,
            "original_size": f"{result['original_size'][0]} × {result['original_size'][1]} px",
            "inference_time_ms": result["inference_time_ms"],
            "device": result["device"],
            "model_type": result.get("model_type", model_type).upper(),
        })

    except ModelNotLoadedError as e:
        return jsonify({"success": False, "error": str(e)}), 503
    except Exception as e:
        return jsonify({"success": False, "error": f"Image processing failed: {str(e)}"}), 500


@app.route("/colorize-sample", methods=["POST"])
def colorize_sample():
    """
    Colorize one of the pre-loaded demo images without requiring manual upload.
    """
    data = request.get_json() or {}
    model_type = data.get("model_type", "eccv16").lower()
    sample_name = secure_filename(data.get("sample_name", ""))

    if model_type in ("unet", "custom") and not colorizer.is_loaded:
        if not colorizer.load_checkpoint():
            return jsonify({
                "success": False,
                "error": "Custom GAN checkpoint not found. Please run training first, or choose Pretrained ECCV16.",
            }), 503

    if not sample_name:
        return jsonify({"success": False, "error": "No sample specified."}), 400

    sample_path = os.path.join(config.SAMPLES_DIR, sample_name)
    if not os.path.exists(sample_path):
        return jsonify({"success": False, "error": "Selected sample does not exist."}), 404

    try:
        unique_id = uuid.uuid4().hex
        ext = sample_name.rsplit(".", 1)[1].lower() if "." in sample_name else "jpg"
        input_filename = f"sample_{unique_id}.{ext}"
        output_filename = f"sample_{unique_id}_colorized.png"
        comparison_filename = f"sample_{unique_id}_comparison.png"

        input_path = os.path.join(config.UPLOAD_DIR, input_filename)
        output_path = os.path.join(config.OUTPUT_DIR, output_filename)
        comparison_path = os.path.join(config.OUTPUT_DIR, comparison_filename)

        # Copy sample to uploads directory
        shutil.copyfile(sample_path, input_path)

        # Run colorizer
        result = colorizer.colorize(
            image_path=input_path,
            output_path=output_path,
            preserve_resolution=True,
            save_comparison=True,
            comparison_path=comparison_path,
            model_type=model_type,
        )

        return jsonify({
            "success": True,
            "input_url": url_for("static", filename=f"uploads/{input_filename}"),
            "output_url": url_for("static", filename=f"outputs/{output_filename}"),
            "comparison_url": url_for("static", filename=f"outputs/{comparison_filename}"),
            "download_filename": output_filename,
            "original_size": f"{result['original_size'][0]} × {result['original_size'][1]} px",
            "inference_time_ms": result["inference_time_ms"],
            "device": result["device"],
            "model_type": result.get("model_type", model_type).upper(),
        })
    except Exception as e:
        return jsonify({"success": False, "error": f"Failed to colorize sample: {str(e)}"}), 500


@app.route("/download/<path:filename>")
def download_image(filename):
    """
    Safe file download endpoint.
    Prevents path traversal attacks using secure basename extraction.
    """
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(config.OUTPUT_DIR, safe_filename)

    if not os.path.exists(file_path):
        return jsonify({"error": "Requested file not found."}), 404

    return send_from_directory(
        config.OUTPUT_DIR,
        safe_filename,
        as_attachment=True,
        download_name=f"colorized_{safe_filename}",
    )

if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)
