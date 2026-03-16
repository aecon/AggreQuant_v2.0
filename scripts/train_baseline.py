"""Train baseline UNet on annotated aggregate data.

Usage:
    conda run -n AggreQuant python scripts/train_baseline.py

Prerequisite: symlinks must exist in training_output/baseline/symlinks/
    (images/ and masks/ with matching filenames). Create once with:
    see project devlog or run the symlink commands manually.

This script:
1. Extracts 192x192 non-overlapping patches from 19 annotated images
2. Splits patches 80/20 train/val (shuffled, seed=42)
3. Trains a baseline UNet with weighted BCE (pos_weight=7.5)
4. Saves best checkpoint to training_output/baseline/checkpoints/best.pt
"""

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
CHECKPOINT_DIR = TRAINING_ROOT / "baseline" / "checkpoints"

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

PATCH_SIZE = 192
BATCH_SIZE = 16
EPOCHS = 200
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 20
POS_WEIGHT = 7.5       # matches old weighted BCE (agg=7.5, bkg=1.0)
VAL_SPLIT = 0.2
SEED = 42
NUM_WORKERS = 4


def main():
    logger.info("=" * 60)
    logger.info("Training baseline UNet on aggregate data")
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
    criterion = DiceBCELoss(alpha=0.0, beta=1.0, pos_weight=POS_WEIGHT)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6,
    )

    # Metrics: dice and IoU computed on sigmoid outputs
    metrics = {
        "dice": lambda preds, targets: dice_score(preds, targets),
        "iou": lambda preds, targets: iou_score(preds, targets),
        "precision": lambda preds, targets: precision_score(preds, targets),
        "recall": lambda preds, targets: recall_score(preds, targets),
    }

    # 5. Train
    logger.info("Step 5: Training...")
    logger.info(f"  Loss: BCE with pos_weight={POS_WEIGHT}")
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
        checkpoint_dir=CHECKPOINT_DIR,
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
    logger.info(f"Checkpoint: {CHECKPOINT_DIR / 'best.pt'}")
    logger.info(f"History: {CHECKPOINT_DIR / 'history.json'}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
