"""
Training loop for segmentation models.

This module provides a standard PyTorch training loop with checkpointing,
metrics logging, and support for deep supervision.

Example:
    >>> from aggrequant.nn.training.trainer import Trainer
    >>> trainer = Trainer(model, train_loader, val_loader, criterion, optimizer)
    >>> history = trainer.fit(epochs=100)
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Tuple, Union
from dataclasses import dataclass, field
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from aggrequant.common.logging import get_logger
from aggrequant.nn.utils import get_device

logger = get_logger(__name__)

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


@dataclass
class TrainingHistory:
    """Container for training history."""
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    train_metrics: Dict[str, List[float]] = field(default_factory=dict)
    val_metrics: Dict[str, List[float]] = field(default_factory=dict)
    learning_rates: List[float] = field(default_factory=list)
    best_val_loss: float = float('inf')
    best_epoch: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'learning_rates': self.learning_rates,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
        }

    def save(self, path: Union[str, Path]) -> None:
        """Save history to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'TrainingHistory':
        """Load history from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        history = cls()
        history.train_loss = data['train_loss']
        history.val_loss = data['val_loss']
        history.train_metrics = data['train_metrics']
        history.val_metrics = data['val_metrics']
        history.learning_rates = data['learning_rates']
        history.best_val_loss = data['best_val_loss']
        history.best_epoch = data['best_epoch']
        return history


class Trainer:
    """Training loop for segmentation models.

    Arguments:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data (optional)
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler (optional)
        device: Device to train on (default: auto-detect)
        checkpoint_dir: Directory for saving checkpoints
        metrics: Dictionary of metric functions {name: fn(pred, target)}
        verbose: Print training progress

    Example:
        >>> model = UNet(in_channels=1, out_channels=1)
        >>> criterion = DiceBCELoss()
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        >>> trainer = Trainer(
        ...     model=model,
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ...     criterion=criterion,
        ...     optimizer=optimizer,
        ... )
        >>> history = trainer.fit(epochs=100)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        device: Optional[torch.device] = None,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        metrics: Optional[Dict[str, Callable]] = None,
        verbose: bool = True,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics = metrics or {}
        self.verbose = verbose

        # Auto-detect device
        self.device = get_device(device)

        # Move model to device
        self.model = self.model.to(self.device)

        # Setup checkpoint directory
        if checkpoint_dir is not None:
            self.checkpoint_dir = Path(checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.checkpoint_dir = None

        # Training state
        self.current_epoch = 0
        self.history = TrainingHistory()

        if verbose:
            logger.info(f"Training on device: {self.device}")
            params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Model parameters: {params:,}")

    def train_epoch(self) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch.

        Returns:
            Tuple of (average_loss, metrics_dict)
        """
        self.model.train()
        total_loss = 0.0
        metric_totals = {name: 0.0 for name in self.metrics}
        n_batches = 0

        # Progress bar
        if HAS_TQDM and self.verbose:
            loader = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        else:
            loader = self.train_loader

        for batch in loader:
            images, masks = batch
            images = images.to(self.device)
            masks = masks.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)

            # Compute loss
            loss = self.criterion(outputs, masks)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Accumulate loss
            total_loss += loss.item()
            n_batches += 1

            # Compute metrics (on main output if deep supervision)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            with torch.no_grad():
                preds = torch.sigmoid(outputs)
                for name, metric_fn in self.metrics.items():
                    metric_totals[name] += metric_fn(preds, masks).item()

            # Update progress bar
            if HAS_TQDM and self.verbose:
                loader.set_postfix(loss=loss.item())

        avg_loss = total_loss / n_batches
        avg_metrics = {name: total / n_batches for name, total in metric_totals.items()}

        return avg_loss, avg_metrics

    @torch.no_grad()
    def validate_epoch(self) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch.

        Returns:
            Tuple of (average_loss, metrics_dict)
        """
        if self.val_loader is None:
            return 0.0, {}

        self.model.eval()
        total_loss = 0.0
        metric_totals = {name: 0.0 for name in self.metrics}
        n_batches = 0

        for batch in self.val_loader:
            images, masks = batch
            images = images.to(self.device)
            masks = masks.to(self.device)

            # Forward pass
            outputs = self.model(images)

            # Compute loss
            loss = self.criterion(outputs, masks)
            total_loss += loss.item()
            n_batches += 1

            # Compute metrics (on main output if deep supervision)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            preds = torch.sigmoid(outputs)
            for name, metric_fn in self.metrics.items():
                metric_totals[name] += metric_fn(preds, masks).item()

        avg_loss = total_loss / n_batches
        avg_metrics = {name: total / n_batches for name, total in metric_totals.items()}

        return avg_loss, avg_metrics

    def fit(
        self,
        epochs: int,
        early_stopping_patience: Optional[int] = None,
        save_best_only: bool = True,
    ) -> TrainingHistory:
        """Train the model.

        Arguments:
            epochs: Number of epochs to train
            early_stopping_patience: Stop if no improvement for N epochs (None to disable).
                Requires val_loader — ignored if no validation data is provided.
            save_best_only: Only save checkpoint when validation loss improves

        Returns:
            TrainingHistory object with training metrics
        """
        if self.verbose:
            logger.info(f"Training for {epochs} epochs...")

        patience_counter = 0
        start_time = time.time()

        for epoch in range(epochs):
            self.current_epoch = epoch + 1

            # Train
            train_loss, train_metrics = self.train_epoch()
            self.history.train_loss.append(train_loss)
            for name, value in train_metrics.items():
                if name not in self.history.train_metrics:
                    self.history.train_metrics[name] = []
                self.history.train_metrics[name].append(value)

            # Validate
            val_loss, val_metrics = self.validate_epoch()
            self.history.val_loss.append(val_loss)
            for name, value in val_metrics.items():
                if name not in self.history.val_metrics:
                    self.history.val_metrics[name] = []
                self.history.val_metrics[name].append(value)

            # Step scheduler (ReduceLROnPlateau needs the monitored metric)
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Record LR after scheduler step
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history.learning_rates.append(current_lr)

            # Check for improvement (only meaningful with validation)
            is_best = val_loss < self.history.best_val_loss
            if is_best:
                self.history.best_val_loss = val_loss
                self.history.best_epoch = self.current_epoch
                patience_counter = 0
            else:
                patience_counter += 1

            # Log progress
            if self.verbose:
                val_str = f", val_loss: {val_loss:.4f}" if self.val_loader else ""
                metrics_str = ", ".join(
                    f"{k}: {v:.4f}" for k, v in val_metrics.items()
                )
                if metrics_str:
                    metrics_str = ", " + metrics_str
                logger.info(
                    f"Epoch {self.current_epoch}/{epochs} - "
                    f"loss: {train_loss:.4f}{val_str}{metrics_str} - "
                    f"lr: {current_lr:.2e}"
                )

            # Save checkpoint and history
            if self.checkpoint_dir is not None:
                self.history.save(self.checkpoint_dir / "history.json")
                if save_best_only and is_best:
                    self.save_checkpoint("best.pt")
                elif not save_best_only:
                    self.save_checkpoint(f"epoch_{self.current_epoch:03d}.pt")

            # Early stopping (only with validation data)
            if (
                early_stopping_patience is not None
                and self.val_loader is not None
                and patience_counter >= early_stopping_patience
            ):
                if self.verbose:
                    logger.info(
                        f"Early stopping at epoch {self.current_epoch} "
                        f"(no improvement for {early_stopping_patience} epochs)"
                    )
                break

        # Training complete
        total_time = time.time() - start_time
        if self.verbose:
            logger.info(
                f"Training complete in {total_time:.1f}s. "
                f"Best val_loss: {self.history.best_val_loss:.4f} "
                f"at epoch {self.history.best_epoch}"
            )

        # Save final checkpoint and history
        if self.checkpoint_dir is not None:
            self.save_checkpoint("final.pt")
            self.history.save(self.checkpoint_dir / "history.json")

        return self.history

    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint.

        Arguments:
            filename: Checkpoint filename
        """
        if self.checkpoint_dir is None:
            return

        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.history.best_val_loss,
        }

        # Save model architecture config so checkpoints are self-contained
        if hasattr(self.model, 'get_config'):
            checkpoint['model_config'] = self.model.get_config()

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, self.checkpoint_dir / filename)

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """Load model checkpoint.

        Arguments:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.history.best_val_loss = checkpoint['best_val_loss']

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.verbose:
            logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
