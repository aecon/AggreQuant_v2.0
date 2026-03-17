"""Plot training curves from history.json.

Usage:
    python scripts/plot_training.py training_output/baseline/checkpoints/history.json
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def load_history(path):
    with open(path) as f:
        return json.load(f)


def _plot_metric_pair(ax, epochs, history, primary, secondary, best_epoch=None):
    """Plot a pair of metrics: primary in black, secondary in blue."""
    train_metrics = history.get("train_metrics", {})
    val_metrics = history.get("val_metrics", {})

    if primary in train_metrics:
        ax.plot(epochs, train_metrics[primary], color="black", linestyle="-", label=f"Train {primary}")
    if primary in val_metrics:
        ax.plot(epochs, val_metrics[primary], color="black", linestyle="--", label=f"Val {primary}")
    if secondary in train_metrics:
        ax.plot(epochs, train_metrics[secondary], color="blue", linestyle="-", label=f"Train {secondary}")
    if secondary in val_metrics:
        ax.plot(epochs, val_metrics[secondary], color="blue", linestyle="--", label=f"Val {secondary}")

    if best_epoch:
        ax.axvline(best_epoch, color="gray", linestyle="--", alpha=0.5)


def plot_training(history, save_path=None):
    epochs = range(1, len(history["train_loss"]) + 1)
    best_epoch = history.get("best_epoch", None)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # --- Loss ---
    ax = axes[0, 0]
    ax.plot(epochs, history["train_loss"], color="black", linestyle="-", label="Train")
    ax.plot(epochs, history["val_loss"], color="black", linestyle="--", label="Val")
    if best_epoch:
        ax.axvline(best_epoch, color="gray", linestyle="--", alpha=0.5, label=f"Best (epoch {best_epoch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Dice (black) & IoU (blue) ---
    ax = axes[0, 1]
    _plot_metric_pair(ax, epochs, history, "dice", "iou", best_epoch)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("Dice & IoU")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Recall (black) & Precision (blue) ---
    ax = axes[1, 0]
    _plot_metric_pair(ax, epochs, history, "recall", "precision", best_epoch)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("Precision & Recall")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Learning rate ---
    ax = axes[1, 1]
    if history.get("learning_rates"):
        ax.plot(epochs, history["learning_rates"], color="black")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
    else:
        ax.set_visible(False)

    fig.suptitle(f"Training History ({len(list(epochs))} epochs)", fontsize=14)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")

    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/plot_training.py <history.json> [output.png]")
        sys.exit(1)

    history_path = Path(sys.argv[1])
    save_path = Path(sys.argv[2]) if len(sys.argv) > 2 else history_path.with_suffix(".png")

    history = load_history(history_path)
    plot_training(history, save_path=save_path)
