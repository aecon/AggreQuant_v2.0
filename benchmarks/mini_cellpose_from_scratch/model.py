"""Simplified UNet with style vector for cell segmentation.

A minimal reimplementation of the Cellpose network architecture:
  - Encoder: 4 levels with 2 conv blocks each + max-pool
  - Bottleneck: global average pool -> L2-normalize -> style vector
  - Decoder: upsample + skip concat + 2 conv blocks + style injection
  - Output: 1x1 conv -> 3 channels (flow_y, flow_x, cell_prob)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two consecutive (Conv2d -> BatchNorm2d -> ReLU) layers."""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=pad)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=pad)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class StyleBlock(nn.Module):
    """Projects the style vector and adds it to feature maps.

    Linear(style_dim -> feat_channels), then broadcast-add across spatial dims.
    """

    def __init__(self, style_channels, feat_channels):
        super().__init__()
        self.linear = nn.Linear(style_channels, feat_channels)

    def forward(self, x, style):
        # style: (B, style_channels) -> (B, feat_channels) -> (B, feat_channels, 1, 1)
        s = self.linear(style).unsqueeze(-1).unsqueeze(-1)
        return x + s


class MiniCellposeUNet(nn.Module):
    """Simplified Cellpose-style UNet with style vector.

    Encoder levels: [32, 64, 128, 256]
    Bottleneck: global avg pool -> L2 norm -> style (256-dim)
    Decoder levels: [128, 64, 32] with skip connections + style
    Output head: 1x1 conv -> 3 channels

    Args:
        in_channels: Number of input channels (1 for FarRed only).
        nbase: List of feature sizes for encoder levels.
    """

    def __init__(self, in_channels=1, nbase=None):
        super().__init__()
        if nbase is None:
            nbase = [32, 64, 128, 256]

        self.pool = nn.MaxPool2d(2)
        style_dim = nbase[-1]

        # Encoder
        self.enc1 = ConvBlock(in_channels, nbase[0])
        self.enc2 = ConvBlock(nbase[0], nbase[1])
        self.enc3 = ConvBlock(nbase[1], nbase[2])
        self.enc4 = ConvBlock(nbase[2], nbase[3])

        # Decoder (upsample + concat skip + conv + style)
        self.up3 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec3 = ConvBlock(nbase[3] + nbase[2], nbase[2])
        self.style3 = StyleBlock(style_dim, nbase[2])

        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec2 = ConvBlock(nbase[2] + nbase[1], nbase[1])
        self.style2 = StyleBlock(style_dim, nbase[1])

        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec1 = ConvBlock(nbase[1] + nbase[0], nbase[0])
        self.style1 = StyleBlock(style_dim, nbase[0])

        # Output head
        self.output = nn.Conv2d(nbase[0], 3, kernel_size=1)

    def forward(self, x):
        """Forward pass.

        Args:
            x: (B, C, H, W) input tensor.

        Returns:
            out: (B, 3, H, W) with [flow_y, flow_x, cell_prob_logit].
            style: (B, style_dim) style vector.
        """
        # Encoder
        e1 = self.enc1(x)          # (B, 32, H, W)
        e2 = self.enc2(self.pool(e1))  # (B, 64, H/2, W/2)
        e3 = self.enc3(self.pool(e2))  # (B, 128, H/4, W/4)
        e4 = self.enc4(self.pool(e3))  # (B, 256, H/8, W/8)

        # Style vector: global average pool + L2 normalize
        style = F.adaptive_avg_pool2d(e4, 1).flatten(1)  # (B, 256)
        style = F.normalize(style, p=2, dim=1)

        # Decoder
        d3 = self.up3(e4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d3 = self.style3(d3, style)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d2 = self.style2(d2, style)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        d1 = self.style1(d1, style)

        out = self.output(d1)  # (B, 3, H, W)
        return out, style
