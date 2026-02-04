"""Training loop for segmentation models.

This module provides a standard PyTorch training loop with checkpointing,
metrics logging, and support for deep supervision.

Author: Athena Economides

Example:
    >>> from aggrequant.nn.training.trainer import Trainer
    >>> trainer = Trainer(model, train_loader, val_loader, criterion, optimizer)
    >>> history = trainer.fit(epochs=100)
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Tuple, Union
from dataclasses import dataclass, field
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

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
        debug: Print detailed debug information

    Example:
        >>> model = ModularUNet(in_channels=1, out_channels=1)
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
        debug: bool = False,
    ) -> None:
        me = "Trainer.__init__"

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics = metrics or {}
        self.verbose = verbose
        self.debug = debug

        # Auto-detect device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

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
            print(f"({me}) Training on device: {self.device}")
            params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"({me}) Model parameters: {params:,}")

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
            early_stopping_patience: Stop if no improvement for N epochs (None to disable)
            save_best_only: Only save checkpoint when validation loss improves

        Returns:
            TrainingHistory object with training metrics
        """
        me = "Trainer.fit"

        if self.verbose:
            print(f"({me}) Training for {epochs} epochs...")

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

            # Learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history.learning_rates.append(current_lr)

            # Step scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # Check for improvement
            is_best = val_loss < self.history.best_val_loss
            if is_best:
                self.history.best_val_loss = val_loss
                self.history.best_epoch = self.current_epoch
                patience_counter = 0
            else:
                patience_counter += 1

            # Print progress
            if self.verbose:
                val_str = f", val_loss: {val_loss:.4f}" if self.val_loader else ""
                metrics_str = ", ".join(
                    f"{k}: {v:.4f}" for k, v in val_metrics.items()
                )
                if metrics_str:
                    metrics_str = ", " + metrics_str
                print(
                    f"Epoch {self.current_epoch}/{epochs} - "
                    f"loss: {train_loss:.4f}{val_str}{metrics_str} - "
                    f"lr: {current_lr:.2e}"
                )

            # Save checkpoint
            if self.checkpoint_dir is not None:
                if save_best_only and is_best:
                    self.save_checkpoint("best.pt")
                elif not save_best_only:
                    self.save_checkpoint(f"epoch_{self.current_epoch:03d}.pt")

            # Early stopping
            if early_stopping_patience is not None and patience_counter >= early_stopping_patience:
                if self.verbose:
                    print(
                        f"({me}) Early stopping at epoch {self.current_epoch} "
                        f"(no improvement for {early_stopping_patience} epochs)"
                    )
                break

        # Training complete
        total_time = time.time() - start_time
        if self.verbose:
            print(
                f"({me}) Training complete in {total_time:.1f}s. "
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

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, self.checkpoint_dir / filename)

        if self.debug:
            print(f"Saved checkpoint: {self.checkpoint_dir / filename}")

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """Load model checkpoint.

        Arguments:
            path: Path to checkpoint file
        """
        me = "Trainer.load_checkpoint"

        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.history.best_val_loss = checkpoint['best_val_loss']

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.verbose:
            print(f"({me}) Loaded checkpoint from epoch {self.current_epoch}")


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    criterion: Optional[nn.Module] = None,
    optimizer: Optional[Optimizer] = None,
    epochs: int = 100,
    lr: float = 1e-4,
    checkpoint_dir: Optional[Union[str, Path]] = None,
    early_stopping_patience: Optional[int] = 10,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Tuple[nn.Module, TrainingHistory]:
    """Convenience function for training a model.

    Arguments:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function (default: DiceBCELoss)
        optimizer: Optimizer (default: Adam with given lr)
        epochs: Number of training epochs
        lr: Learning rate (if optimizer not provided)
        checkpoint_dir: Directory for checkpoints
        early_stopping_patience: Early stopping patience
        device: Training device
        verbose: Print progress

    Returns:
        Tuple of (trained_model, history)

    Example:
        >>> model = ModularUNet()
        >>> model, history = train_model(model, train_loader, val_loader)
    """
    from .losses import DiceBCELoss

    # Default criterion
    if criterion is None:
        criterion = DiceBCELoss()

    # Default optimizer
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        checkpoint_dir=checkpoint_dir,
        device=device,
        verbose=verbose,
    )

    # Train
    history = trainer.fit(
        epochs=epochs,
        early_stopping_patience=early_stopping_patience,
    )

    return model, history


__all__ = [
    "Trainer",
    "TrainingHistory",
    "train_model",
]
