# ChromaGAN: Black and White to Color Image Conversion Using GAN

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Flask 3.x](https://img.shields.io/badge/Flask-3.x-black.svg)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests Passing](https://img.shields.io/badge/Tests-16%20Passed-brightgreen.svg)]()

> A full-stack Deep Learning mini-project built for third-year Computer Science / AIML students. Implements a Conditional Generative Adversarial Network (cGAN) with a U-Net Generator and PatchGAN Discriminator in CIE $L^*a^*b^*$ color space, complete with a modern Flask web application, interactive comparison slider, and high-resolution luminance preservation.

---

## Table of Contents
1. [Project Abstract](#1-project-abstract)
2. [Problem Statement](#2-problem-statement)
3. [Motivation and Objectives](#3-motivation-and-objectives)
4. [Existing System Limitations](#4-existing-system-limitations)
5. [Proposed System](#5-proposed-system)
6. [GAN Theory and Working Principle](#6-gan-theory-and-working-principle)
7. [Generator and Discriminator Architecture](#7-generator-and-discriminator-architecture)
8. [Dataset Preparation](#8-dataset-preparation)
9. [Installation Instructions](#9-installation-instructions)
10. [Model Training Instructions](#10-model-training-instructions)
11. [Web Application Execution](#11-web-application-execution)
12. [Testing Procedure](#12-testing-procedure)
13. [Evaluation Metrics](#13-evaluation-metrics)
14. [Results and Visual Showcase](#14-results-and-visual-showcase)
15. [Advantages and Limitations](#15-advantages-and-limitations)
16. [Future Scope](#16-future-scope)
17. [References](#17-references)

---

## 1. Project Abstract
Image colorization is an intrinsically ill-posed and multimodal computer vision challenge: for any single grayscale photograph, countless valid color assignments are plausible. This project develops an end-to-end Deep Learning system utilizing a **Conditional Generative Adversarial Network (cGAN)** built on the Pix2Pix framework. 

Operating in the perceptual **CIE $L^*a^*b^*$ color space**, the model accepts the single-channel Lightness ($L^*$) component and synthesizes the missing two-channel chrominance ($a^*, b^*$) coordinates. The generator uses an 8-stage **U-Net** architecture with symmetric skip connections, coupled with a $70 \times 70$ convolutional **PatchGAN discriminator** that penalizes unrealistic local color patches. To eliminate blurriness when processing high-resolution user photographs, we introduce a **high-resolution luminance preservation** pipeline that pairs the native full-resolution Lightness channel with bicubically upsampled chrominance predictions. The complete system includes an interactive, responsive Flask web application featuring drag-and-drop uploads, live inference, a before-and-after split slider, and high-resolution export.

---

## 2. Problem Statement
Given a single-channel grayscale image $I_{\text{gray}} \in \mathbb{R}^{H \times W \times 1}$, synthesize an authentic, visually compelling three-channel colorized photograph $\hat{I}_{\text{color}} \in \mathbb{R}^{H \times W \times 3}$.

The primary technical hurdle is **multimodality**. For instance, an apple can be red or green, and clothing can be virtually any color. Naive regression systems minimizing pixel-wise Mean Squared Error (MSE) predict the conditional mean of all possible color states, leading to desaturated, dull, and grayish artifacts. Our goal is to leverage adversarial training to produce vibrant, plausible, and high-frequency color distributions.

---

## 3. Motivation and Objectives
- **Cultural & Historical Preservation:** Restore historical monochrome photographs, archival collections, and classic family portraits with realistic color.
- **Multimodal Deep Learning Exploration:** Apply generative adversarial learning to image-to-image translation tasks.
- **Objectives:**
  1. Construct a PyTorch cGAN architecture using U-Net and PatchGAN in CIE $L^*a^*b^*$ space.
  2. Implement dual-network optimization combining LSGAN adversarial loss with $L_1$ reconstruction loss ($\lambda = 100.0$).
  3. Ensure sharp, unblurred outputs by preserving original high-resolution luminance.
  4. Build a modern, accessible web interface with drag-and-drop, instant sample testing, and an interactive split slider.
  5. Provide a reproducible dataset and automated test suite.

---

## 4. Existing System Limitations
- **Manual Recoloring:** Requires digital artists to hand-paint masks and color layers in Adobe Photoshop; takes hours per image.
- **Scribble-Based Algorithms:** Requires users to manually place colored strokes to propagate colors across boundaries.
- **Direct RGB Regression:** CNNs trained to minimize Mean Squared Error (MSE) directly on RGB values suffer from mode collapse, producing sepia or murky brown outputs.
- **Resolution Loss:** Standard image translation networks downscale input images to $256 \times 256$, irreversibly destroying photographic resolution and sharpness.

---

## 5. Proposed System
Our system addresses these limitations through four key innovations:
1. **Perceptual CIE $L^*a^*b^*$ Decoupling:** Separates brightness ($L^*$) from color ($a^*, b^*$). The grayscale input is directly preserved as the true $L^*$ channel.
2. **U-Net Generator with Skip Connections:** Concatenates encoder feature maps directly to decoder layers, preserving fine structural boundaries and preventing color bleeding.
3. **PatchGAN Discriminator:** Evaluates local $70 \times 70$ receptive fields, penalizing unnatural local color distributions.
4. **Hybrid High-Resolution Upsampling:** The generator infers at $256 \times 256$, and its continuous $a^*b^*$ prediction is bicubically interpolated back to the original image dimensions and merged with the native full-resolution Lightness channel.

```
+------------------+       +-------------------+       +-----------------------+
|  Grayscale Input | ----> |  Extract L* (Lab) | ----> |  U-Net 256 Generator  |
+------------------+       +-------------------+       +-----------------------+
         |                           |                             |
         | (Full Resolution)         | (256x256)                   v (Predicted a*b*)
         |                           |                 +-----------------------+
         |                           |                 | Bicubic Upsampling    |
         |                           |                 +-----------------------+
         v                           v                             |
+------------------------------------------------------------------+-----------+
| Stack: Original High-Res L*  +  Upscaled Predicted a*b*                      |
+------------------------------------------------------------------------------+
                                     |
                                     v
                       +---------------------------+
                       | Convert CIE Lab to sRGB   |
                       +---------------------------+
                                     |
                                     v
                       +---------------------------+
                       | High-Res Colorized Image  |
                       +---------------------------+
```

---

## 6. GAN Theory and Working Principle
A Conditional Generative Adversarial Network (cGAN) models the mapping from an observed image $x$ and random noise $z$ to an output image $y$:

$$G: \{x, z\} \to y$$

The objective function combines conditional adversarial loss with an $L_1$ reconstruction distance:

$$\mathcal{L}_{\text{cGAN}}(G, D) = \mathbb{E}_{x, y} \left[ \log D(x, y) \right] + \mathbb{E}_{x, z} \left[ \log (1 - D(x, G(x, z))) \right]$$

$$\mathcal{L}_{L_1}(G) = \mathbb{E}_{x, y, z} \left[ \| y - G(x, z) \|_1 \right]$$

$$\mathcal{L}_{\text{total}} = \arg \min_G \max_D \left( \mathcal{L}_{\text{cGAN}}(G, D) + \lambda \mathcal{L}_{L_1}(G) \right), \quad \lambda = 100.0$$

In our implementation, we adopt **Least Squares GAN (LSGAN)** which replaces the logarithmic cross-entropy with Mean Squared Error to eliminate vanishing gradients and stabilize color convergence:

$$\mathcal{L}_{D} = \frac{1}{2} \mathbb{E} \left[ (D(x, y) - 1)^2 \right] + \frac{1}{2} \mathbb{E} \left[ (D(x, G(x)))^2 \right]$$

$$\mathcal{L}_{G} = \mathbb{E} \left[ (D(x, G(x)) - 1)^2 \right] + \lambda \| y - G(x) \|_1$$

---

## 7. Generator and Discriminator Architecture

### Generator (U-Net 256)
- **Input:** $[B, 1, 256, 256]$ (Normalized Lightness $\tilde{L} \in [-1.0, 1.0]$)
- **Encoder:** 8 Convolutional downsampling blocks with stride 2 and LeakyReLU(0.2).
- **Bottleneck:** $[B, 512, 1, 1]$ compact latent representation.
- **Decoder:** 8 Transposed Convolutional upsampling blocks with ReLU, Dropout(0.5) in early layers, and channel-wise skip concatenation.
- **Output Layer:** $\text{ConvTranspose2d}(128 \to 2) \to \text{Tanh}()$, outputting $[\tilde{a}, \tilde{b}] \in [-1.0, 1.0]$.

### Discriminator (70x70 PatchGAN)
- **Input:** $[B, 3, 256, 256]$ (Concatenation of Lightness $L^*$ and real/generated $a^*b^*$).
- **Layers:** 5 Convolutional blocks with kernel size 4, stride 2 (and stride 1 in the final stages).
- **Output:** $[B, 1, 30, 30]$ patch matrix. Each output unit classifies whether a corresponding $70 \times 70$ receptive field patch is real or synthetic.

---

## 8. Dataset Preparation
The system supports any folder of standard natural RGB photographs (e.g., JPEG, PNG, WEBP).

### Recommended Public Datasets
1. **Places365-Standard:** High-diversity scenes and landscapes: [Places365 Dataset](http://places2.csail.mit.edu/)
2. **COCO 2017 Val/Train:** Diverse natural objects and humans: [COCO Dataset](https://cocodataset.org/)
3. **Flickr Natural Landscapes / Kaggle Landscape Images:** Clean landscape collections.

### Folder Structure
Organize your dataset as follows:
```
data/
  ├── train/
  │     ├── img_001.jpg
  │     └── img_002.jpg
  └── val/
        ├── img_101.jpg
        └── img_102.jpg
```
*Note: If separate `train/` and `val/` directories are not provided, `train.py` automatically performs a deterministic random split (85% train / 15% validation) without data leakage.*

---

## 9. Installation Instructions

### Windows (PowerShell)
```powershell
# 1. Clone repository
git clone https://github.com/your-username/Colorization-GAN.git
cd "Colorization-GAN"

# 2. Create and activate a Python virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Linux / macOS
```bash
# 1. Clone and navigate to repository
git clone https://github.com/your-username/Colorization-GAN.git
cd Colorization-GAN

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install requirements
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 10. Model Training Instructions

### Local Training (CPU or Local NVIDIA GPU)
```powershell
# Option A: Quick Smoke Test (verifies full pipeline in seconds with synthetic data)
python train.py --smoke_test

# Option B: Full Training on Custom Dataset
python train.py --epochs 40 --batch_size 16 --lambda_l1 100.0 --data_dir data
```

### Resume Training from Checkpoint
```powershell
python train.py --resume checkpoints/latest_checkpoint.pth --epochs 60
```

### Google Colab GPU Training (Free T4 / A100)
1. Open Google Colab: `https://colab.research.google.com/`.
2. Upload `colab_training.ipynb` included in this repository.
3. Select **Runtime > Change runtime type > T4 GPU**.
4. Run all cells to install dependencies, train with GPU acceleration, and download the resulting `best_generator.pth`.

---

## 11. Web Application Execution
Once a generator checkpoint exists in `checkpoints/best_generator.pth`:

```powershell
# Start the Flask development server
python app.py
```

Open your browser and navigate to:
```
http://127.0.0.1:5000
```

### Web App Features
- **Drag & Drop Upload:** Accepts PNG, JPG, JPEG, WEBP up to 16MB.
- **Demo Gallery:** Click any of the pre-loaded sample images for instant one-click demonstration.
- **Interactive Split Slider:** Drag the circular divider horizontally to smoothly reveal monochrome input vs. GAN colorization.
- **Side-by-Side View:** Toggle between split slider and dual-panel view.
- **High-Res Download:** Exports the synthesized photograph in full native resolution.

---

## 12. Testing Procedure
Run the automated test suite with PyTest:

```powershell
python -m pytest tests/ -v
```

### Test Suite Coverage
- `tests/test_preprocessing.py`: Validates RGB $\leftrightarrow$ Lab mathematical round-trips, tensor bounds, and bicubic upsampling.
- `tests/test_model.py`: Validates Generator output shape $[B, 2, 256, 256]$, PatchGAN output shape $[B, 1, 30, 30]$, LSGAN loss, and optimization steps.
- `tests/test_inference.py`: Validates end-to-end colorization with resolution preservation and missing checkpoint error handling.
- `tests/test_app.py`: Validates Flask routes (`/`, `/status`, `/colorize`, file validation, and `/download`).

---

## 13. Evaluation Metrics
For quantitative benchmarking against paired ground-truth color photographs:
- **Peak Signal-to-Noise Ratio (PSNR):** Measures fidelity in decibels (dB). Higher values ($>25\text{ dB}$) indicate higher fidelity.
- **Structural Similarity Index (SSIM):** Evaluates structural, contrast, and luminance consistency in $[-1.0, 1.0]$. Values $>0.80$ reflect strong structural preservation.
- **Mean Absolute Error (MAE):** Average $L_1$ error across $a^*$ and $b^*$ chrominance channels.

---

## 14. Results and Visual Showcase
After training, visual results and learning curves are saved to `results/`:
- `results/loss_plot.png`: Dual-panel loss graphs showing Generator vs. Discriminator total losses alongside GAN adversarial vs. $L_1$ loss components.
- `results/training_history.json`: Machine-readable epoch loss logs.
- `results/epoch_XXX_sample_Y.png`: Side-by-side grids comparing `[Grayscale Input | GAN Synthesized | Ground Truth]`.

---

## 15. Advantages and Limitations

### Advantages
- **Automatic & Non-Parametric:** Requires no manual color hints, reference palettes, or user scribbles.
- **Preserved Native Sharpness:** Retains 100% of input lightness detail via $L^*$ channel bypass.
- **CIE Lab Decoupling:** Eliminates the muddy brown mode collapse typical of naive RGB regression.
- **Production-Ready Web App:** Clean dark navy glassmorphism UI, drag-and-drop, interactive split slider, and download capability.

### Limitations
- **Historical Ambiguity:** Vintage photographs have no objective ground truth; the model predicts plausible colors based on learned natural priors, but cannot know historical color choices (e.g., fabric dyes).
- **Context-Free Objects:** Symmetrical or isolated synthetic objects with no environmental context may default to desaturated tones.

---

## 16. Future Scope
1. **Text-Prompt Guidance:** Incorporate CLIP embeddings to allow users to guide color synthesis (e.g., *"autumn foliage"*, *"golden hour sunset"*).
2. **Video Colorization:** Integrate spatio-temporal consistency losses to colorize vintage black-and-white video footage without inter-frame flickering.
3. **Edge Optimization:** Quantize the U-Net model with ONNX Runtime or TensorRT for real-time mobile and browser execution.

---

## 17. References
1. **Isola, P., Zhu, J. Y., Zhou, T., & Efros, A. A. (2017).** Image-to-Image Translation with Conditional Adversarial Networks. *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 1125–1134.
2. **Goodfellow, I., Pouget-Abadie, J., Mirza, M., et al. (2014).** Generative Adversarial Nets. *Advances in Neural Information Processing Systems (NIPS)*, pp. 2672–2680.
3. **Zhang, R., Isola, P., & Efros, A. A. (2016).** Colorful Image Colorization. *European Conference on Computer Vision (ECCV)*, pp. 649–666.
4. **Mao, X., Li, Q., Xie, H., Lau, R. Y., Wang, Z., & Smolley, S. P. (2017).** Least Squares Generative Adversarial Networks. *IEEE International Conference on Computer Vision (ICCV)*, pp. 2794–2802.
5. **Ronneberger, O., Fischer, P., & Brox, T. (2015).** U-Net: Convolutional Networks for Biomedical Image Segmentation. *MICCAI 2015*, pp. 234–241.
6. **PyTorch Framework Documentation:** [https://pytorch.org/docs/](https://pytorch.org/docs/)
7. **Flask Web Framework:** [https://flask.palletsprojects.com/](https://flask.palletsprojects.com/)

---

## Project Structure Overview
```
Colorization-GAN/
├── app.py                      # Flask web application & REST endpoints
├── train.py                    # Dual-network cGAN training pipeline
├── inference.py                # Standalone inference & color reconstruction
├── config.py                   # Central hyperparameters & paths
├── requirements.txt            # Python dependencies
├── README.md                   # Comprehensive project documentation
├── PROJECT_REPORT.md           # College mini-project formal report
├── colab_training.ipynb        # Google Colab GPU training notebook
├── .gitignore                  # Git ignore rules
│
├── models/
│   ├── __init__.py
│   ├── generator.py            # U-Net Generator with skip connections
│   ├── discriminator.py        # 70x70 PatchGAN Discriminator
│   └── gan.py                  # cGAN system, LSGAN loss, weight initialization
│
├── utils/
│   ├── __init__.py
│   ├── dataset.py              # PyTorch Dataset, DataLoaders, augmentation
│   ├── image_utils.py          # CIE Lab conversions, bicubic upscaling, grids
│   └── metrics.py              # PSNR, SSIM, MAE quantitative metrics
│
├── templates/
│   └── index.html              # Modern dark navy UI with split comparison slider
│
├── static/
│   ├── css/
│   │   └── style.css           # Glassmorphism dark theme styling
│   ├── js/
│   │   └── script.js           # Drag-and-drop & split slider controller
│   ├── uploads/                # User uploaded images
│   ├── outputs/                # Colorized generated images
│   └── samples/                # Demo gallery photos
│
├── data/
│   ├── train/                  # Training images
│   └── val/                    # Validation images
│
├── checkpoints/                # Saved weights (best_generator.pth)
├── results/                    # Loss curves & validation sample comparisons
│
└── tests/
    ├── test_preprocessing.py   # CIE Lab roundtrip & normalization tests
    ├── test_model.py           # Network dimensions & loss tests
    ├── test_inference.py       # Inference pipeline & error tests
    └── test_app.py             # Flask endpoints & validation tests
```
