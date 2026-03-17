"""Train baseline UNet on annotated aggregate data.

Usage:
    python scripts/train_baseline.py --name baseline
    python scripts/train_baseline.py --name dice_bce_pw3 --alpha 0.3 --beta 0.7 --pos-weight 3.0
    python scripts/train_baseline.py --name bce_pw3 --alpha 0.0 --beta 1.0 --pos-weight 3.0

Prerequisite: symlinks must exist in training_output/symlinks/
    (images/ and masks/ with matching filenames).
"""

import argparse
from pathlib import Path

import torch

from aggrequant.nn.architectures.registry import create_model
from aggrequant.nn.datatools.dataset import extract_patches, create_dataloaders
from aggrequant.nn.datatools.augmentation import get_training_augmentation
from aggrequant.nn.training.trainer import Trainer
from aggrequant.nn.training.losses import DiceBCELoss
from aggrequant.nn.evaluation.metrics import dice_score, iou_score, precision_score, recall_score
from aggrequant.common.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

TRAINING_ROOT = Path(__file__).resolve().parent.parent / "training_output"
SYMLINK_DIR = TRAINING_ROOT / "symlinks"
PATCH_DIR = TRAINING_ROOT / "patches"

# ---------------------------------------------------------------------------
# Fixed hyperparameters
# ---------------------------------------------------------------------------

PATCH_SIZE = 192
BATCH_SIZE = 16
EPOCHS = 200
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 20
VAL_SPLIT = 0.2
SEED = 42
NUM_WORKERS = 4


def parse_args():
    parser = argparse.ArgumentParser(description="Train baseline UNet")
    parser.add_argument("--name", type=str, required=True,
                        help="Experiment name (output goes to training_output/<name>/)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Dice loss weight (default: 0.5)")
    parser.add_argument("--beta", type=float, default=0.5,
                        help="BCE loss weight (default: 0.5)")
    parser.add_argument("--pos-weight", type=float, default=None,
                        help="BCE positive class weight (default: None)")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_dir = TRAINING_ROOT / args.name / "checkpoints"

    logger.info("=" * 60)
    logger.info(f"Training baseline UNet — experiment: {args.name}")
    logger.info("=" * 60)

    # Check symlinks exist
    img_dir = SYMLINK_DIR / "images"
    mask_dir = SYMLINK_DIR / "masks"
    if not img_dir.exists() or len(list(img_dir.glob("*.tif"))) == 0:
        raise FileNotFoundError(
            f"No symlinks found in {SYMLINK_DIR}. "
            "Create them first (see script docstring)."
        )
    n_images = len(list(img_dir.glob("*.tif")))

    # 1. Extract patches
    logger.info(f"Step 1: Extracting {PATCH_SIZE}x{PATCH_SIZE} patches...")
    if (PATCH_DIR / "images").exists() and len(list((PATCH_DIR / "images").glob("*.tif"))) > 0:
        n_patches = len(list((PATCH_DIR / "images").glob("*.tif")))
        logger.info(f"Patches already exist ({n_patches}), skipping extraction")
    else:
        n_patches = extract_patches(
            image_dir=SYMLINK_DIR / "images",
            mask_dir=SYMLINK_DIR / "masks",
            output_dir=PATCH_DIR,
            patch_size=PATCH_SIZE,
        )
    logger.info(f"Total patches: {n_patches} from {n_images} images")

    # 2. Create dataloaders
    logger.info("Step 2: Creating dataloaders...")
    augmentation = get_training_augmentation(
        scale_limit=(1.0, 1.0),  # no zoom
    )
    train_loader, val_loader = create_dataloaders(
        patch_dir=PATCH_DIR,
        train_transform=augmentation,
        val_transform=None,
        val_split=VAL_SPLIT,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        seed=SEED,
    )

    # 3. Create model
    logger.info("Step 3: Creating baseline UNet...")
    model = create_model("baseline", in_channels=1, out_channels=1)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")

    # 4. Setup loss, optimizer, scheduler
    criterion = DiceBCELoss(alpha=args.alpha, beta=args.beta, pos_weight=args.pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6,
    )

    metrics = {
        "dice": lambda preds, targets: dice_score(preds, targets),
        "iou": lambda preds, targets: iou_score(preds, targets),
        "precision": lambda preds, targets: precision_score(preds, targets),
        "recall": lambda preds, targets: recall_score(preds, targets),
    }

    # 5. Train
    loss_desc = f"Dice(alpha={args.alpha}) + BCE(beta={args.beta}"
    if args.pos_weight is not None:
        loss_desc += f", pos_weight={args.pos_weight}"
    loss_desc += ")"

    logger.info("Step 5: Training...")
    logger.info(f"  Loss: {loss_desc}")
    logger.info(f"  Optimizer: Adam, lr={LEARNING_RATE}")
    logger.info(f"  Scheduler: ReduceLROnPlateau (factor=0.5, patience=10, min_lr=1e-6)")
    logger.info(f"  Epochs: {EPOCHS}, early stopping patience: {EARLY_STOPPING_PATIENCE}")
    logger.info(f"  Batch size: {BATCH_SIZE}")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_dir=checkpoint_dir,
        metrics=metrics,
        verbose=True,
    )

    history = trainer.fit(
        epochs=EPOCHS,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        save_best_only=True,
    )

    # 6. Summary
    logger.info("=" * 60)
    logger.info(f"Best val_loss: {history.best_val_loss:.4f} at epoch {history.best_epoch}")
    if "dice" in history.val_metrics:
        best_dice = history.val_metrics["dice"][history.best_epoch - 1]
        logger.info(f"Best val_dice: {best_dice:.4f}")
    if "iou" in history.val_metrics:
        best_iou = history.val_metrics["iou"][history.best_epoch - 1]
        logger.info(f"Best val_iou: {best_iou:.4f}")
    logger.info(f"Checkpoint: {checkpoint_dir / 'best.pt'}")
    logger.info(f"History: {checkpoint_dir / 'history.json'}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
