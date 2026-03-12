"""Tests for nn/ architecture building blocks."""

import torch
import pytest

from aggrequant.nn.architectures.blocks.conv import SingleConv, DoubleConv
from aggrequant.nn.architectures.blocks.residual import ResidualBlock, BottleneckResidualBlock
from aggrequant.nn.architectures.blocks.attention import AttentionGate, MultiHeadAttentionGate
from aggrequant.nn.architectures.blocks.se import SEBlock, SEConvBlock, SEResidualBlock
from aggrequant.nn.architectures.blocks.cbam import ChannelAttention, SpatialAttention, CBAM
from aggrequant.nn.architectures.blocks.eca import ECABlock
from aggrequant.nn.architectures.blocks.convnext import ConvNeXtBlock, LayerNorm2d
from aggrequant.nn.architectures.blocks.aspp import ASPP, ASPPBridge, LightASPP


# --- Fixtures ---

@pytest.fixture(scope="module")
def x_1ch():
    """Input tensor with 1 channel (B=1, C=1, H=64, W=64)."""
    return torch.randn(1, 1, 64, 64)

@pytest.fixture(scope="module")
def x_32ch():
    """Input tensor with 32 channels."""
    return torch.randn(1, 32, 64, 64)

@pytest.fixture(scope="module")
def x_64ch():
    """Input tensor with 64 channels."""
    return torch.randn(1, 64, 64, 64)


# --- SingleConv ---

def test_single_conv_output_shape(x_1ch):
    block = SingleConv(1, 32)
    out = block(x_1ch)
    assert out.shape == (1, 32, 64, 64)

def test_single_conv_same_channels(x_32ch):
    block = SingleConv(32, 32)
    out = block(x_32ch)
    assert out.shape == x_32ch.shape


# --- DoubleConv ---

@pytest.mark.parametrize("in_ch,out_ch", [(1, 32), (32, 64), (64, 64)])
def test_double_conv_output_shape(in_ch, out_ch):
    x = torch.randn(1, in_ch, 64, 64)
    block = DoubleConv(in_ch, out_ch)
    out = block(x)
    assert out.shape == (1, out_ch, 64, 64)

def test_double_conv_mid_channels():
    block = DoubleConv(1, 64, mid_channels=32)
    out = block(torch.randn(1, 1, 64, 64))
    assert out.shape == (1, 64, 64, 64)


# --- ResidualBlock ---

@pytest.mark.parametrize("in_ch,out_ch", [(32, 32), (32, 64)])
def test_residual_block_output_shape(in_ch, out_ch):
    x = torch.randn(1, in_ch, 64, 64)
    block = ResidualBlock(in_ch, out_ch)
    out = block(x)
    assert out.shape == (1, out_ch, 64, 64)

def test_residual_block_skip_identity():
    """When in_ch == out_ch, skip connection is Identity."""
    block = ResidualBlock(32, 32)
    assert isinstance(block.skip, torch.nn.Identity)

def test_residual_block_skip_conv():
    """When in_ch != out_ch, skip connection is a 1x1 conv."""
    block = ResidualBlock(32, 64)
    assert not isinstance(block.skip, torch.nn.Identity)


# --- BottleneckResidualBlock ---

def test_bottleneck_residual_output_shape():
    block = BottleneckResidualBlock(64, 64)
    out = block(torch.randn(1, 64, 32, 32))
    assert out.shape == (1, 64, 32, 32)

def test_bottleneck_residual_channel_change():
    block = BottleneckResidualBlock(32, 64)
    out = block(torch.randn(1, 32, 32, 32))
    assert out.shape == (1, 64, 32, 32)


# --- AttentionGate ---

def test_attention_gate_output_shape():
    gate = AttentionGate(gate_channels=64, skip_channels=32)
    g = torch.randn(1, 64, 16, 16)
    x = torch.randn(1, 32, 32, 32)
    out = gate(g, x)
    assert out.shape == (1, 32, 32, 32)

def test_attention_gate_same_spatial():
    """When gate and skip have same spatial size."""
    gate = AttentionGate(gate_channels=64, skip_channels=32)
    g = torch.randn(1, 64, 32, 32)
    x = torch.randn(1, 32, 32, 32)
    out = gate(g, x)
    assert out.shape == x.shape


# --- MultiHeadAttentionGate ---

def test_multihead_attention_gate_output_shape():
    gate = MultiHeadAttentionGate(gate_channels=64, skip_channels=32, num_heads=4)
    g = torch.randn(1, 64, 16, 16)
    x = torch.randn(1, 32, 32, 32)
    out = gate(g, x)
    assert out.shape == (1, 32, 32, 32)


# --- SEBlock ---

def test_se_block_output_shape(x_32ch):
    block = SEBlock(channels=32)
    out = block(x_32ch)
    assert out.shape == x_32ch.shape

def test_se_block_small_channels():
    """Reduction shouldn't make bottleneck zero."""
    block = SEBlock(channels=4, reduction=16)
    out = block(torch.randn(1, 4, 16, 16))
    assert out.shape == (1, 4, 16, 16)


# --- SEConvBlock ---

def test_se_conv_block_output_shape():
    block = SEConvBlock(32, 64)
    out = block(torch.randn(1, 32, 64, 64))
    assert out.shape == (1, 64, 64, 64)


# --- SEResidualBlock ---

@pytest.mark.parametrize("in_ch,out_ch", [(32, 32), (32, 64)])
def test_se_residual_block_output_shape(in_ch, out_ch):
    block = SEResidualBlock(in_ch, out_ch)
    out = block(torch.randn(1, in_ch, 32, 32))
    assert out.shape == (1, out_ch, 32, 32)


# --- ChannelAttention ---

def test_channel_attention_output_shape(x_32ch):
    ca = ChannelAttention(32)
    out = ca(x_32ch)
    assert out.shape == (1, 32, 1, 1)

def test_channel_attention_values_in_0_1(x_32ch):
    ca = ChannelAttention(32)
    out = ca(x_32ch)
    assert out.min() >= 0.0 and out.max() <= 1.0


# --- SpatialAttention ---

def test_spatial_attention_output_shape(x_32ch):
    sa = SpatialAttention()
    out = sa(x_32ch)
    assert out.shape == (1, 1, 64, 64)

def test_spatial_attention_values_in_0_1(x_32ch):
    sa = SpatialAttention()
    out = sa(x_32ch)
    assert out.min() >= 0.0 and out.max() <= 1.0


# --- CBAM ---

def test_cbam_output_shape(x_32ch):
    block = CBAM(32)
    out = block(x_32ch)
    assert out.shape == x_32ch.shape


# --- ECABlock ---

def test_eca_block_output_shape(x_32ch):
    block = ECABlock(channels=32)
    out = block(x_32ch)
    assert out.shape == x_32ch.shape

def test_eca_block_explicit_kernel():
    block = ECABlock(channels=64, kernel_size=5)
    out = block(torch.randn(1, 64, 32, 32))
    assert out.shape == (1, 64, 32, 32)


# --- LayerNorm2d ---

def test_layer_norm_2d_output_shape(x_32ch):
    norm = LayerNorm2d(32)
    out = norm(x_32ch)
    assert out.shape == x_32ch.shape


# --- ConvNeXtBlock ---

def test_convnext_block_same_channels(x_32ch):
    block = ConvNeXtBlock(32, 32)
    out = block(x_32ch)
    assert out.shape == x_32ch.shape

def test_convnext_block_channel_change():
    block = ConvNeXtBlock(32, 64)
    out = block(torch.randn(1, 32, 64, 64))
    assert out.shape == (1, 64, 64, 64)


# --- ASPP ---

def test_aspp_output_shape():
    block = ASPP(64, 32)
    out = block(torch.randn(1, 64, 16, 16))
    assert out.shape == (1, 32, 16, 16)

def test_aspp_default_out_channels():
    """out_channels defaults to in_channels."""
    block = ASPP(64)
    out = block(torch.randn(1, 64, 16, 16))
    assert out.shape == (1, 64, 16, 16)


# --- ASPPBridge ---

def test_aspp_bridge_output_shape():
    block = ASPPBridge(64, 128)
    out = block(torch.randn(1, 64, 8, 8))
    assert out.shape == (1, 128, 8, 8)


# --- LightASPP ---

def test_light_aspp_output_shape():
    block = LightASPP(64, 32)
    out = block(torch.randn(1, 64, 32, 32))
    assert out.shape == (1, 32, 32, 32)

def test_light_aspp_default_out_channels():
    block = LightASPP(64)
    out = block(torch.randn(1, 64, 32, 32))
    assert out.shape == (1, 64, 32, 32)


# --- Gradient flow ---

def test_double_conv_gradient_flows():
    block = DoubleConv(1, 32)
    x = torch.randn(1, 1, 32, 32, requires_grad=True)
    out = block(x)
    out.sum().backward()
    assert x.grad is not None
    assert torch.all(torch.isfinite(x.grad))

def test_residual_block_gradient_flows():
    block = ResidualBlock(32, 64)
    x = torch.randn(1, 32, 32, 32, requires_grad=True)
    out = block(x)
    out.sum().backward()
    assert x.grad is not None
    assert torch.all(torch.isfinite(x.grad))

def test_convnext_block_gradient_flows():
    block = ConvNeXtBlock(32, 32)
    x = torch.randn(1, 32, 32, 32, requires_grad=True)
    out = block(x)
    out.sum().backward()
    assert x.grad is not None
    assert torch.all(torch.isfinite(x.grad))
