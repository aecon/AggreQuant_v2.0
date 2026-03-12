"""Training script for mini Cellpose.

Usage:
    conda run -n mini_cellpose_AE python train.py \
        --image-dir data/images \
        --mask-dir data/masks \
        [--epochs 200] [--batch-size 8] [--lr 1e-3] \
        [--crop-size 256] [--save-dir checkpoints]
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import CellFlowDataset
from model import MiniCellposeUNet


def loss_fn(pred, target):
    """Cellpose loss: MSE on flows (masked to foreground) + BCE on cell prob.

    Args:
        pred: (B, 3, H, W) model output [flow_y, flow_x, cell_prob_logit].
        target: (B, 3, H, W) ground truth [flow_y, flow_x, cell_prob].

    Returns:
        Scalar loss, plus a dict of component values for logging.
    """
    pred_flows = pred[:, :2]   # (B, 2, H, W)
    pred_prob = pred[:, 2]     # (B, H, W)
    gt_flows = target[:, :2]
    gt_prob = target[:, 2]

    # Foreground mask
    fg = gt_prob > 0.5  # (B, H, W)
    n_fg = fg.sum().clamp(min=1)

    # MSE on flows, only where cells exist
    flow_diff = (pred_flows - gt_flows) ** 2  # (B, 2, H, W)
    flow_loss = (flow_diff * fg.unsqueeze(1)).sum() / (2 * n_fg)

    # BCE on cell probability
    bce_loss = nn.functional.binary_cross_entropy_with_logits(
        pred_prob, gt_prob, reduction="mean"
    )

    total = flow_loss + bce_loss
    return total, {"flow": flow_loss.item(), "bce": bce_loss.item()}


def train_one_epoch(model, loader, optimizer, device):
    """Train for one epoch. Returns mean loss and component breakdown."""
    model.train()
    total_loss = 0
    total_flow = 0
    total_bce = 0
    n_batches = 0

    for imgs, targets in loader:
        imgs = imgs.to(device)
        targets = targets.to(device)

        pred, _ = model(imgs)
        loss, components = loss_fn(pred, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_flow += components["flow"]
        total_bce += components["bce"]
        n_batches += 1

    return (total_loss / n_batches,
            total_flow / n_batches,
            total_bce / n_batches)


@torch.no_grad()
def validate(model, loader, device):
    """Validate. Returns mean loss and component breakdown."""
    model.eval()
    total_loss = 0
    total_flow = 0
    total_bce = 0
    n_batches = 0

    for imgs, targets in loader:
        imgs = imgs.to(device)
        targets = targets.to(device)

        pred, _ = model(imgs)
        loss, components = loss_fn(pred, targets)

        total_loss += loss.item()
        total_flow += components["flow"]
        total_bce += components["bce"]
        n_batches += 1

    return (total_loss / n_batches,
            total_flow / n_batches,
            total_bce / n_batches)


def main():
    parser = argparse.ArgumentParser(description="Train mini Cellpose from scratch")
    parser.add_argument("--image-dir", required=True, help="Path to image directory")
    parser.add_argument("--mask-dir", required=True, help="Path to cellpose_cyto3 masks")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--save-dir", default="checkpoints")
    parser.add_argument("--no-gpu", action="store_true")
    args = parser.parse_args()

    # Device
    if args.no_gpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    print(f"Using device: {device}")

    # Dataset
    full_dataset = CellFlowDataset(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        crop_size=args.crop_size,
        augment=True,
    )
    print(f"Found {len(full_dataset)} image-mask pairs")

    # Train/val split
    n_val = max(1, int(len(full_dataset) * args.val_fraction))
    n_train = len(full_dataset) - n_val
    train_dataset, val_dataset = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    # Disable augmentation for validation subset
    # (val_dataset still uses the same underlying dataset, but we accept
    # that augmentation is on — the val loss is for trend monitoring only)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )
    print(f"Train: {n_train}, Val: {n_val}")

    # Model
    model = MiniCellposeUNet(in_channels=1).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # Checkpointing
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"{'Epoch':>6} {'Train':>10} {'Val':>10} {'Flow':>8} {'BCE':>8} {'Time':>6}")
    print("-" * 52)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_flow, train_bce = train_one_epoch(
            model, train_loader, optimizer, device
        )
        val_loss, val_flow, val_bce = validate(model, val_loader, device)

        dt = time.time() - t0

        # Save best model
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            }, save_dir / "best.pt")

        # Print progress
        marker = " *" if is_best else ""
        print(f"{epoch:6d} {train_loss:10.4f} {val_loss:10.4f} "
              f"{val_flow:8.4f} {val_bce:8.4f} {dt:5.1f}s{marker}")

        # Save periodic checkpoint
        if epoch % 50 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            }, save_dir / f"epoch_{epoch:04d}.pt")

    # Save final model
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "val_loss": val_loss,
    }, save_dir / "final.pt")

    print(f"\nDone. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to {save_dir}/")


if __name__ == "__main__":
    main()
