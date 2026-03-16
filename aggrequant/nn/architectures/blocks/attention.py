"""
Attention gates for Attention U-Net architecture.

This module implements attention gates as described in:
"Attention U-Net: Learning Where to Look for the Pancreas"
(Oktay et al., 2018)

Attention gates help the network focus on relevant features in skip
connections by using the decoder features as a gating signal.
"""

import torch
import torch.nn as nn


class AttentionGate(nn.Module):
    """Attention gate for skip connections in Attention U-Net.

    The attention gate learns to suppress irrelevant regions in the skip
    connection features while highlighting salient features useful for
    the specific task.

    The gating signal (g) comes from the decoder path and guides attention
    on the skip connection features (x).

    Arguments:
        gate_channels: Number of channels in gating signal (from decoder)
        skip_channels: Number of channels in skip connection (from encoder)
        inter_channels: Number of intermediate channels for attention computation
            (default: half of skip_channels)

    Returns:
        Attention-weighted skip features of shape (B, skip_channels, H, W)

    Example:
        >>> attention = AttentionGate(gate_channels=256, skip_channels=128)
        >>> g = torch.randn(1, 256, 32, 32)  # decoder features (coarse)
        >>> x = torch.randn(1, 128, 64, 64)  # skip features (fine)
        >>> out = attention(g, x)
        >>> out.shape
        torch.Size([1, 128, 64, 64])

    References:
        Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas"
        https://arxiv.org/abs/1804.03999
    """

    def __init__(
        self,
        gate_channels: int,
        skip_channels: int,
        inter_channels: int = None,
    ) -> None:
        super().__init__()

        if inter_channels is None:
            inter_channels = skip_channels // 2
            if inter_channels == 0:  # handles the edge case where skip_channels = 1
                inter_channels = 1

        # Transform gating signal to intermediate space
        self.W_g = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(inter_channels),
        )

        # Transform skip connection to intermediate space
        self.W_x = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(inter_channels),
        )

        # Attention coefficient computation
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Apply attention gate to skip connection features.

        Arguments:
            g: Gating signal from decoder path (B, gate_channels, H_g, W_g)
            x: Skip connection features from encoder (B, skip_channels, H_x, W_x)
                Note: H_x, W_x are typically 2x H_g, W_g

        Returns:
            Attention-weighted features of shape (B, skip_channels, H_x, W_x)
        """
        # Transform gating signal
        g1 = self.W_g(g)

        # Upsample gating signal to match skip connection spatial size
        g1 = nn.functional.interpolate(
            g1, size=x.shape[2:], mode="bilinear", align_corners=False
        )

        # Transform skip connection
        x1 = self.W_x(x)

        # Combine and compute attention coefficients
        combined = self.relu(g1 + x1)
        attention = self.psi(combined)

        # Apply attention to skip features
        return x * attention


class MultiHeadAttentionGate(nn.Module):
    """Multi-head attention gate for enhanced feature selection.

    Extension of the standard attention gate using multiple attention heads,
    similar to multi-head attention in transformers. Each head attends to
    different aspects of the features.

    Arguments:
        gate_channels: Number of channels in gating signal
        skip_channels: Number of channels in skip connection
        num_heads: Number of attention heads (default: 4)

    Returns:
        Attention-weighted skip features of shape (B, skip_channels, H, W)

    Example:
        >>> attention = MultiHeadAttentionGate(256, 128, num_heads=4)
        >>> g = torch.randn(1, 256, 32, 32)
        >>> x = torch.randn(1, 128, 64, 64)
        >>> out = attention(g, x)
        >>> out.shape
        torch.Size([1, 128, 64, 64])
    """

    def __init__(
        self,
        gate_channels: int,
        skip_channels: int,
        num_heads: int = 4,
    ) -> None:
        super().__init__()

        self.num_heads = num_heads
        assert (
            skip_channels % num_heads == 0
        ), f"skip_channels ({skip_channels}) must be divisible by num_heads ({num_heads})"

        head_channels = skip_channels // num_heads

        # Create attention gates for each head
        self.attention_heads = nn.ModuleList(
            [
                AttentionGate(gate_channels, head_channels, head_channels // 2)
                for _ in range(num_heads)
            ]
        )

        # Split and merge projections
        self.split_proj = nn.Conv2d(skip_channels, skip_channels, kernel_size=1)
        self.merge_proj = nn.Conv2d(skip_channels, skip_channels, kernel_size=1)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-head attention gate.

        Arguments:
            g: Gating signal from decoder (B, gate_channels, H_g, W_g)
            x: Skip features from encoder (B, skip_channels, H_x, W_x)

        Returns:
            Attention-weighted features of shape (B, skip_channels, H_x, W_x)
        """
        # Project and split into heads
        x_proj = self.split_proj(x)
        head_size = x_proj.shape[1] // self.num_heads

        # Apply attention per head
        head_outputs = []
        for i, attn in enumerate(self.attention_heads):
            x_head = x_proj[:, i * head_size : (i + 1) * head_size, :, :]
            head_outputs.append(attn(g, x_head))

        # Concatenate and merge heads
        concat = torch.cat(head_outputs, dim=1)
        return self.merge_proj(concat)
