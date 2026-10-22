"""
Training Pipeline for B&W to Color Image Conversion Using Conditional GAN.

Features:
- Dual network optimization (U-Net Generator + PatchGAN Discriminator)
- Adversarial loss + L1 color reconstruction loss
- Validation evaluation with MAE / PSNR tracking
- Best model checkpointing & periodic checkpoints
- Resume training support
- Loss curve plotting with Matplotlib
- Epoch visual sample generation (Grayscale | Predicted | Ground Truth)
- Smoke test mode for immediate end-to-end pipeline verification
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Tuple
import numpy as np
import torch
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")  # Headless backend for server/CLI environments
import matplotlib.pyplot as plt
from PIL import Image

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from models.gan import ColorizationGAN
from utils.dataset import create_dataloaders, generate_sample_dataset, get_image_files
from utils.image_utils import lab_tensors_to_rgb, create_comparison_grid
from utils.metrics import compute_mae, compute_psnr


def parse_args():
    parser = argparse.ArgumentParser(description="Train Conditional GAN for Image Colorization")
    parser.add_argument("--data_dir", type=str, default=config.DATA_DIR, help="Path to data directory")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr_g", type=float, default=config.LEARNING_RATE_G, help="Generator learning rate")
    parser.add_argument("--lr_d", type=float, default=config.LEARNING_RATE_D, help="Discriminator learning rate")
    parser.add_argument("--lambda_l1", type=float, default=config.LAMBDA_L1, help="L1 reconstruction loss weight")
    parser.add_argument("--img_size", type=int, default=config.IMG_SIZE, help="Image spatial resolution")
    parser.add_argument("--checkpoint_dir", type=str, default=config.CHECKPOINT_DIR, help="Directory to save weights")
    parser.add_argument("--results_dir", type=str, default=config.RESULTS_DIR, help="Directory to save sample plots")
    parser.add_argument("--save_freq", type=int, default=config.SAVE_FREQ, help="Checkpoint saving frequency (epochs)")
    parser.add_argument("--sample_freq", type=int, default=config.SAMPLE_FREQ, help="Sample generation frequency")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--smoke_test", action="store_true", help="Run 1 quick epoch on small synthetic dataset")
    return parser.parse_args()


def plot_training_history(history: Dict[str, List[float]], save_path: str):
    """Plot and save training loss curves."""
    epochs = range(1, len(history["loss_g"]) + 1)

    plt.figure(figsize=(12, 5))

    # Subplot 1: Generator & Discriminator Total Losses
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["loss_g"], label="Generator Loss", color="#6366F1", linewidth=2)
    plt.plot(epochs, history["loss_d"], label="Discriminator Loss", color="#EF4444", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Total Generator vs Discriminator Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    # Subplot 2: Generator Loss Components (Adversarial vs L1)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["loss_g_gan"], label="GAN Adversarial Loss", color="#06B6D4", linewidth=1.8)
    plt.plot(epochs, history["loss_g_l1"], label="Weighted L1 Loss", color="#10B981", linewidth=1.8)
    if len(history.get("val_mae", [])) == len(epochs):
        plt.plot(epochs, history["val_mae"], label="Val MAE (ab)", color="#F59E0B", linestyle=":", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss Component Value")
    plt.title("Loss Components Breakdown")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_epoch_sample_visualization(
    model: ColorizationGAN,
    val_loader,
    epoch: int,
    results_dir: str,
    device: torch.device,
    max_samples: int = 3,
):
    """
    Renders visual side-by-side comparison images:
    [Grayscale L Input] | [GAN Predicted Color] | [Ground Truth Color]
    """
    model.generator.eval()
    samples_created = 0

    with torch.no_grad():
        for l_batch, ab_batch in val_loader:
            l_batch = l_batch.to(device)
            ab_batch = ab_batch.to(device)

            fake_ab_batch = model.generator(l_batch)

            for i in range(min(l_batch.size(0), max_samples - samples_created)):
                l_single = l_batch[i].cpu()
                fake_ab_single = fake_ab_batch[i].cpu()
                real_ab_single = ab_batch[i].cpu()

                # Reconstruct grayscale image (using ab=0)
                zero_ab = torch.zeros_like(fake_ab_single)
                gray_rgb = lab_tensors_to_rgb(l_single, zero_ab)

                # Reconstruct GAN predicted color image
                pred_rgb = lab_tensors_to_rgb(l_single, fake_ab_single)

                # Reconstruct ground truth color image
                real_rgb = lab_tensors_to_rgb(l_single, real_ab_single)

                # Create comparison grid
                grid = create_comparison_grid(gray_rgb, pred_rgb, real_rgb)

                out_path = os.path.join(results_dir, f"epoch_{epoch:03d}_sample_{samples_created+1}.png")
                Image.fromarray(grid).save(out_path)
                samples_created += 1

            if samples_created >= max_samples:
                break

    model.generator.train()


def evaluate_validation(model: ColorizationGAN, val_loader, device: torch.device) -> Tuple[float, float]:
    """
    Evaluates generator on validation split.
    Returns:
        (mean_val_mae, mean_val_psnr)
    """
    model.generator.eval()
    mae_list = []
    psnr_list = []

    with torch.no_grad():
        for l_batch, ab_batch in val_loader:
            l_batch = l_batch.to(device)
            ab_batch = ab_batch.to(device)

            fake_ab = model.generator(l_batch)

            # Compute MAE on ab tensor
            mae = torch.mean(torch.abs(fake_ab - ab_batch)).item()
            mae_list.append(mae)

            # Compute PSNR on first sample of batch
            l_s = l_batch[0].cpu()
            f_ab = fake_ab[0].cpu()
            r_ab = ab_batch[0].cpu()

            pred_rgb = lab_tensors_to_rgb(l_s, f_ab)
            real_rgb = lab_tensors_to_rgb(l_s, r_ab)
            psnr_list.append(compute_psnr(real_rgb, pred_rgb))

    model.generator.train()
    return float(np.mean(mae_list)), float(np.mean(psnr_list))


def train():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    # Smoke test settings override
    if args.smoke_test:
        print("[SMOKE TEST MODE ENABLED]")
        args.epochs = 2
        args.batch_size = 2
        args.save_freq = 1
        args.sample_freq = 1
        # Check if dataset exists; if not, generate tiny synthetic dataset
        train_files = get_image_files(os.path.join(args.data_dir, "train"))
        if len(train_files) == 0:
            print("Generating 8 synthetic sample images for smoke test...")
            generate_sample_dataset(args.data_dir, count=8)

    # Check for dataset existence
    train_files = get_image_files(os.path.join(args.data_dir, "train"))
    val_files = get_image_files(os.path.join(args.data_dir, "val"))
    all_files = get_image_files(args.data_dir)

    if len(all_files) == 0:
        print(f"Warning: No images found in {args.data_dir}. Generating synthetic demo dataset...")
        generate_sample_dataset(args.data_dir, count=16)

    # DataLoaders
    print(f"Loading dataset from: {args.data_dir}")
    train_loader, val_loader, n_train, n_val = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
    )
    print(f"Dataset split: {n_train} training samples, {n_val} validation samples.")

    # Model Initialization
    device = config.DEVICE
    print(f"Executing on hardware device: {device.type.upper()}")
    if device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    gan = ColorizationGAN(
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        lambda_l1=args.lambda_l1,
        device=device,
    )

    start_epoch = 1
    best_val_mae = float("inf")

    # Resume training if requested
    if args.resume:
        if os.path.exists(args.resume):
            print(f"Resuming training from checkpoint: {args.resume}")
            start_epoch = gan.load_checkpoint(args.resume, resume_training=True) + 1
            print(f"Resumed at epoch {start_epoch}")
        else:
            print(f"Checkpoint file '{args.resume}' not found. Starting from scratch.")

    # Tracking History
    history = {
        "loss_d": [],
        "loss_g": [],
        "loss_g_gan": [],
        "loss_g_l1": [],
        "val_mae": [],
        "val_psnr": [],
    }

    print("\n" + "=" * 65)
    print("STARTING COLORIZATION GAN TRAINING")
    print(f"Total Epochs:      {args.epochs}")
    print(f"Batch Size:        {args.batch_size}")
    print(f"Lambda L1 Weight:  {args.lambda_l1}")
    print(f"Generator LR:      {args.lr_g}")
    print(f"Discriminator LR:  {args.lr_d}")
    print("=" * 65 + "\n")

    for epoch in range(start_epoch, args.epochs + 1):
        gan.generator.train()
        gan.discriminator.train()

        running_d = 0.0
        running_g = 0.0
        running_g_gan = 0.0
        running_g_l1 = 0.0
        batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=True)
        for l_batch, ab_batch in pbar:
            gan.set_input(l_batch, ab_batch)
            step_losses = gan.optimize_parameters()

            running_d += step_losses["loss_d"]
            running_g += step_losses["loss_g"]
            running_g_gan += step_losses["loss_g_gan"]
            running_g_l1 += step_losses["loss_g_l1"]
            batches += 1

            pbar.set_postfix({
                "D_loss": f"{running_d / batches:.4f}",
                "G_loss": f"{running_g / batches:.4f}",
                "G_L1": f"{running_g_l1 / batches:.4f}",
            })

        # Epoch Metrics
        epoch_d = running_d / max(1, batches)
        epoch_g = running_g / max(1, batches)
        epoch_g_gan = running_g_gan / max(1, batches)
        epoch_g_l1 = running_g_l1 / max(1, batches)

        # Validation evaluation
        val_mae, val_psnr = evaluate_validation(gan, val_loader, device)

        history["loss_d"].append(epoch_d)
        history["loss_g"].append(epoch_g)
        history["loss_g_gan"].append(epoch_g_gan)
        history["loss_g_l1"].append(epoch_g_l1)
        history["val_mae"].append(val_mae)
        history["val_psnr"].append(val_psnr)

        print(
            f"Epoch [{epoch:03d}/{args.epochs:03d}] "
            f"Loss_D: {epoch_d:.4f} | Loss_G: {epoch_g:.4f} "
            f"(GAN: {epoch_g_gan:.4f}, L1: {epoch_g_l1:.4f}) | "
            f"Val MAE: {val_mae:.4f} | Val PSNR: {val_psnr:.2f} dB"
        )

        # Save Best Model Checkpoint
        is_best = val_mae < best_val_mae
        if is_best:
            best_val_mae = val_mae
            gan.save_generator_only(os.path.join(args.checkpoint_dir, "best_generator.pth"))
            gan.save_checkpoint(os.path.join(args.checkpoint_dir, "best_model.pth"), epoch, is_best=True)
            print(f" -> Best model saved with Val MAE: {val_mae:.4f}")

        # Periodic full checkpoint
        if epoch % args.save_freq == 0 or epoch == args.epochs:
            latest_path = os.path.join(args.checkpoint_dir, "latest_checkpoint.pth")
            gan.save_checkpoint(latest_path, epoch, is_best=False)

        # Periodic visual samples
        if epoch % args.sample_freq == 0 or epoch == args.epochs:
            save_epoch_sample_visualization(gan, val_loader, epoch, args.results_dir, device)

        # Update training loss plots and JSON log
        plot_training_history(history, os.path.join(args.results_dir, "loss_plot.png"))
        with open(os.path.join(args.results_dir, "training_history.json"), "w") as f:
            json.dump(history, f, indent=2)

    # Always ensure a deployment generator is present
    best_gen_path = os.path.join(args.checkpoint_dir, "best_generator.pth")
    if not os.path.exists(best_gen_path):
        gan.save_generator_only(best_gen_path)

    print("\n" + "=" * 65)
    print("TRAINING COMPLETE!")
    print(f"Best Generator weights saved to: {best_gen_path}")
    print(f"Loss curves saved to:           {os.path.join(args.results_dir, 'loss_plot.png')}")
    print(f"Visual samples saved to:        {args.results_dir}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    train()
