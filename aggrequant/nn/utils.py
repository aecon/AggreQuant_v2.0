"""PyTorch utility functions."""

from typing import Optional, Union
import torch


def get_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """
    Get PyTorch device, with automatic GPU detection.

    Arguments:
        device: Device specification. Can be:
            - None: Auto-detect (CUDA if available, else CPU)
            - str: Device name ('cuda', 'cpu', 'cuda:0', etc.)
            - torch.device: Use as-is

    Returns:
        torch.device instance

    Example:
        >>> device = get_device()  # Auto-detect
        >>> device = get_device('cuda')  # Force CUDA
        >>> device = get_device('cpu')  # Force CPU
    """
    if device is None:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, torch.device):
        return device
    else:
        return torch.device(device)
