"""Train all 7 UNet architecture variants for the ablation benchmark.

Trains each registry model with identical hyperparameters and the best loss
configuration from the loss comparison study (DiceBCE alpha=0.3, beta=0.7,
pos_weight=3.0). Writes a summary CSV with metrics for all variants.

Usage:
    # Train all 7 variants (single seed):
    python scripts/train_ablation.py

    # Train a subset:
    python scripts/train_ablation.py --models baseline resunet

    # Multi-seed runs for confidence intervals:
    python scripts/train_ablation.py --seeds 42 123 456

    # Resume (skips variants that already have a best.pt checkpoint):
    python scripts/train_ablation.py

Prerequisite: symlinks must exist in training_output/symlinks/
    (images/ and masks/ with matching filenames).
"""

import argparse
import csv
import json
from pathlib import Path

import torch

from aggrequant.nn.architectures.registry import create_model, list_models
from aggrequant.nn.datatools.dataset import extract_patches, create_dataloaders
from aggrequant.nn.datatools.augmentation import get_training_augmentation
from aggrequant.nn.training.trainer import Trainer
from aggrequant.nn.training.losses import DiceBCELoss
from aggrequant.nn.evaluation.metrics import (
    dice_score, iou_score, precision_score, recall_score,
)
from aggrequant.common.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

TRAINING_ROOT = Path(__file__).resolve().parent.parent / "training_output"
SYMLINK_DIR = TRAINING_ROOT / "symlinks"
PATCH_DIR = TRAINING_ROOT / "patches"

# ---------------------------------------------------------------------------
# Fixed hyperparameters (same for all variants)
# ---------------------------------------------------------------------------

PATCH_SIZE = 192
BATCH_SIZE = 16
EPOCHS = 200
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 20
VAL_SPLIT = 0.2
NUM_WORKERS = 4

# Loss: best from loss comparison study
LOSS_ALPHA = 0.3
LOSS_BETA = 0.7
LOSS_POS_WEIGHT = 3.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train UNet architecture variants for ablation benchmark"
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help=f"Model names to train (default: all). Available: {list_models()}",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[42],
        help="Random seeds for train/val split (default: [42]). "
             "Multiple seeds produce separate runs for statistics.",
    )
    parser.add_argument(
        "--skip-existing", action="store_true", default=True,
        help="Skip variants that already have a best.pt checkpoint (default: True)",
    )
    parser.add_argument(
        "--no-skip-existing", action="store_false", dest="skip_existing",
        help="Retrain all variants even if checkpoints exist",
    )
    return parser.parse_args()


def ensure_patches(seed):
    """Extract patches if they don't already exist for this seed."""
    patch_dir = PATCH_DIR / f"seed_{seed}" if seed != 42 else PATCH_DIR

    img_dir = SYMLINK_DIR / "images"
    mask_dir = SYMLINK_DIR / "masks"
    if not img_dir.exists() or len(list(img_dir.glob("*.tif"))) == 0:
        raise FileNotFoundError(
            f"No symlinks found in {SYMLINK_DIR}. "
            "Create images/ and masks/ subdirectories with matching filenames."
        )

    if (patch_dir / "images").exists() and len(list((patch_dir / "images").glob("*.tif"))) > 0:
        n_patches = len(list((patch_dir / "images").glob("*.tif")))
        logger.info(f"Patches already exist ({n_patches}), skipping extraction")
    else:
        n_patches = extract_patches(
            image_dir=img_dir,
            mask_dir=mask_dir,
            output_dir=patch_dir,
            patch_size=PATCH_SIZE,
        )

    return patch_dir, n_patches


def train_one_variant(model_name, seed, patch_dir, skip_existing):
    """Train a single model variant. Returns dict of results or None if skipped."""
    run_name = model_name if seed == 42 else f"{model_name}_seed{seed}"
    checkpoint_dir = TRAINING_ROOT / "ablation" / run_name / "checkpoints"

    # Check for existing checkpoint
    if skip_existing and (checkpoint_dir / "best.pt").exists():
        logger.info(f"[{run_name}] Checkpoint exists, loading results...")
        return load_existing_results(run_name, checkpoint_dir)

    logger.info("=" * 60)
    logger.info(f"[{run_name}] Training {model_name} (seed={seed})")
    logger.info("=" * 60)

    # Dataloaders
    augmentation = get_training_augmentation(scale_limit=(1.0, 1.0))
    train_loader, val_loader = create_dataloaders(
        patch_dir=patch_dir,
        train_transform=augmentation,
        val_transform=None,
        val_split=VAL_SPLIT,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        seed=seed,
    )

    # Model
    model = create_model(model_name, in_channels=1, out_channels=1)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"[{run_name}] Parameters: {n_params:,}")

    # Loss, optimizer, scheduler
    criterion = DiceBCELoss(
        alpha=LOSS_ALPHA, beta=LOSS_BETA, pos_weight=LOSS_POS_WEIGHT,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6,
    )

    metrics = {
        "dice": lambda p, t: dice_score(p, t),
        "iou": lambda p, t: iou_score(p, t),
        "precision": lambda p, t: precision_score(p, t),
        "recall": lambda p, t: recall_score(p, t),
    }

    # Train
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

    # Collect results at best epoch
    best_idx = history.best_epoch - 1
    result = {
        "model": model_name,
        "seed": seed,
        "params": n_params,
        "best_epoch": history.best_epoch,
        "total_epochs": len(history.train_loss),
        "val_loss": history.best_val_loss,
    }
    for metric_name in ["dice", "iou", "precision", "recall"]:
        if metric_name in history.val_metrics:
            result[metric_name] = history.val_metrics[metric_name][best_idx]

    logger.info(
        f"[{run_name}] Done — Dice: {result.get('dice', 'N/A'):.4f}, "
        f"IoU: {result.get('iou', 'N/A'):.4f}, "
        f"Precision: {result.get('precision', 'N/A'):.4f}, "
        f"Recall: {result.get('recall', 'N/A'):.4f}"
    )

    return result


def load_existing_results(run_name, checkpoint_dir):
    """Load results from a previously completed training run."""
    history_path = checkpoint_dir / "history.json"
    if not history_path.exists():
        logger.warning(f"[{run_name}] No history.json found, cannot load results")
        return None

    with open(history_path) as f:
        history = json.load(f)

    best_epoch = history["best_epoch"]
    best_idx = best_epoch - 1

    # Get param count from checkpoint
    checkpoint = torch.load(
        checkpoint_dir / "best.pt", map_location="cpu", weights_only=False,
    )
    model_config = checkpoint.get("model_config", {})
    n_params = checkpoint.get("n_params", 0)

    # Parse model name and seed from run_name
    parts = run_name.rsplit("_seed", 1)
    model_name = parts[0]
    seed = int(parts[1]) if len(parts) > 1 else 42

    result = {
        "model": model_name,
        "seed": seed,
        "params": n_params,
        "best_epoch": best_epoch,
        "total_epochs": len(history["train_loss"]),
        "val_loss": history["best_val_loss"],
    }
    for metric_name in ["dice", "iou", "precision", "recall"]:
        if metric_name in history.get("val_metrics", {}):
            result[metric_name] = history["val_metrics"][metric_name][best_idx]

    logger.info(
        f"[{run_name}] Loaded — Dice: {result.get('dice', 'N/A'):.4f}, "
        f"IoU: {result.get('iou', 'N/A'):.4f}"
    )

    return result


def write_summary_csv(results, output_path):
    """Write ablation results to CSV."""
    if not results:
        return

    fieldnames = [
        "model", "seed", "params", "best_epoch", "total_epochs",
        "val_loss", "dice", "iou", "precision", "recall",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    logger.info(f"Summary CSV written to {output_path}")


def print_summary_table(results):
    """Print results as a formatted table."""
    if not results:
        return

    logger.info("\n" + "=" * 90)
    logger.info("ABLATION RESULTS")
    logger.info("=" * 90)
    header = f"{'Model':<30} {'Params':>10} {'Dice':>7} {'IoU':>7} {'Prec':>7} {'Recall':>7} {'Epoch':>6}"
    logger.info(header)
    logger.info("-" * 90)
    for r in results:
        line = (
            f"{r['model']:<30} "
            f"{r['params']:>10,} "
            f"{r.get('dice', 0):>7.4f} "
            f"{r.get('iou', 0):>7.4f} "
            f"{r.get('precision', 0):>7.4f} "
            f"{r.get('recall', 0):>7.4f} "
            f"{r['best_epoch']:>6}"
        )
        logger.info(line)
    logger.info("=" * 90)


def main():
    args = parse_args()

    model_names = args.models if args.models else list_models()

    # Validate model names
    available = list_models()
    for name in model_names:
        if name not in available:
            raise ValueError(f"Unknown model '{name}'. Available: {available}")

    logger.info(f"Ablation benchmark: {len(model_names)} models × {len(args.seeds)} seeds")
    logger.info(f"Models: {model_names}")
    logger.info(f"Seeds: {args.seeds}")
    logger.info(
        f"Loss: DiceBCE(alpha={LOSS_ALPHA}, beta={LOSS_BETA}, "
        f"pos_weight={LOSS_POS_WEIGHT})"
    )

    all_results = []

    for seed in args.seeds:
        # Ensure patches exist (shared across all models for a given seed)
        patch_dir, n_patches = ensure_patches(seed)
        logger.info(f"Using {n_patches} patches from {patch_dir}")

        for model_name in model_names:
            result = train_one_variant(
                model_name, seed, patch_dir, args.skip_existing,
            )
            if result is not None:
                all_results.append(result)

    # Write summary
    csv_path = TRAINING_ROOT / "ablation" / "ablation_results.csv"
    write_summary_csv(all_results, csv_path)
    print_summary_table(all_results)


if __name__ == "__main__":
    main()
