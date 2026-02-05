"""Unit tests for neural network architectures.

Tests the modular UNet architecture with different configurations.
"""

import pytest
import torch
import torch.nn as nn

from aggrequant.nn.architectures import UNet, ModularUNet
from aggrequant.nn.architectures.blocks import (
    DoubleConv,
    ResidualBlock,
    AttentionGate,
    SEBlock,
    CBAM,
)


class TestBuildingBlocks:
    """Test individual building blocks."""

    def test_double_conv_shape(self):
        """DoubleConv should maintain spatial dimensions."""
        block = DoubleConv(64, 128)
        x = torch.randn(2, 64, 64, 64)
        out = block(x)
        assert out.shape == (2, 128, 64, 64)

    def test_residual_block_shape(self):
        """ResidualBlock should maintain spatial dimensions."""
        block = ResidualBlock(64, 128)
        x = torch.randn(2, 64, 64, 64)
        out = block(x)
        assert out.shape == (2, 128, 64, 64)

    def test_residual_block_same_channels(self):
        """ResidualBlock with same in/out channels uses identity skip."""
        block = ResidualBlock(64, 64)
        x = torch.randn(2, 64, 64, 64)
        out = block(x)
        assert out.shape == (2, 64, 64, 64)

    def test_se_block_shape(self):
        """SEBlock should maintain all dimensions."""
        se = SEBlock(channels=128, reduction=16)
        x = torch.randn(2, 128, 32, 32)
        out = se(x)
        assert out.shape == x.shape

    def test_cbam_shape(self):
        """CBAM should maintain all dimensions."""
        cbam = CBAM(channels=128)
        x = torch.randn(2, 128, 32, 32)
        out = cbam(x)
        assert out.shape == x.shape

    def test_attention_gate_shape(self):
        """AttentionGate should output same shape as skip connection."""
        attn = AttentionGate(gate_channels=256, skip_channels=128)
        g = torch.randn(2, 256, 16, 16)  # coarse
        x = torch.randn(2, 128, 32, 32)  # fine
        out = attn(g, x)
        assert out.shape == x.shape


class TestUNet:
    """Test UNet architecture with different configurations."""

    def test_baseline_forward(self):
        """Baseline UNet should produce correct output shape."""
        model = UNet(in_channels=1, out_channels=1)
        x = torch.randn(2, 1, 256, 256)
        out = model(x)
        assert out.shape == (2, 1, 256, 256)

    def test_residual_forward(self):
        """Residual UNet should produce correct output shape."""
        model = UNet(
            in_channels=1,
            out_channels=1,
            encoder_block="residual",
            decoder_block="residual",
        )
        x = torch.randn(2, 1, 256, 256)
        out = model(x)
        assert out.shape == (2, 1, 256, 256)

    def test_attention_forward(self):
        """UNet with attention gates should produce correct output shape."""
        model = UNet(
            in_channels=1,
            out_channels=1,
            use_attention_gates=True,
        )
        x = torch.randn(2, 1, 256, 256)
        out = model(x)
        assert out.shape == (2, 1, 256, 256)

    def test_se_forward(self):
        """UNet with SE blocks should produce correct output shape."""
        model = UNet(
            in_channels=1,
            out_channels=1,
            use_se=True,
        )
        x = torch.randn(2, 1, 256, 256)
        out = model(x)
        assert out.shape == (2, 1, 256, 256)

    def test_cbam_forward(self):
        """UNet with CBAM should produce correct output shape."""
        model = UNet(
            in_channels=1,
            out_channels=1,
            use_cbam=True,
        )
        x = torch.randn(2, 1, 256, 256)
        out = model(x)
        assert out.shape == (2, 1, 256, 256)

    def test_aspp_bridge_forward(self):
        """UNet with ASPP bridge should produce correct output shape."""
        model = UNet(
            in_channels=1,
            out_channels=1,
            bridge_type="aspp",
        )
        x = torch.randn(2, 1, 256, 256)
        out = model(x)
        assert out.shape == (2, 1, 256, 256)

    def test_se_cbam_mutually_exclusive(self):
        """SE and CBAM should be mutually exclusive."""
        with pytest.raises(ValueError):
            UNet(use_se=True, use_cbam=True)

    def test_deep_supervision_training(self):
        """Deep supervision should return tuple during training."""
        model = UNet(
            in_channels=1,
            out_channels=1,
            use_deep_supervision=True,
        )
        model.train()
        x = torch.randn(2, 1, 256, 256)
        out = model(x)
        assert isinstance(out, tuple)
        main_out, aux_outs = out
        assert main_out.shape == (2, 1, 256, 256)
        assert isinstance(aux_outs, list)
        assert len(aux_outs) == 3  # 4 levels - 1

    def test_deep_supervision_eval(self):
        """Deep supervision should return single tensor during eval."""
        model = UNet(
            in_channels=1,
            out_channels=1,
            use_deep_supervision=True,
        )
        model.eval()
        x = torch.randn(2, 1, 256, 256)
        out = model(x)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (2, 1, 256, 256)

    def test_custom_features(self):
        """UNet should work with custom feature sizes."""
        model = UNet(
            in_channels=1,
            out_channels=1,
            features=[32, 64, 128, 256],
        )
        x = torch.randn(2, 1, 256, 256)
        out = model(x)
        assert out.shape == (2, 1, 256, 256)

    def test_multichannel_input(self):
        """UNet should handle multi-channel input."""
        model = UNet(in_channels=3, out_channels=1)
        x = torch.randn(2, 3, 256, 256)
        out = model(x)
        assert out.shape == (2, 1, 256, 256)

    def test_multichannel_output(self):
        """UNet should handle multi-channel output."""
        model = UNet(in_channels=1, out_channels=3)
        x = torch.randn(2, 1, 256, 256)
        out = model(x)
        assert out.shape == (2, 3, 256, 256)

    def test_non_power_of_two_input(self):
        """UNet should handle non-power-of-two input sizes."""
        model = UNet(in_channels=1, out_channels=1)
        x = torch.randn(2, 1, 200, 200)
        out = model(x)
        assert out.shape == (2, 1, 200, 200)

    def test_count_parameters(self):
        """count_parameters should return positive integer."""
        model = UNet()
        params = model.count_parameters()
        assert params > 0
        assert isinstance(params, int)


class TestUNetConfigurations:
    """Test various UNet module combinations for A/B testing."""

    def test_residual_with_attention(self):
        """Residual + attention gates should work together."""
        model = UNet(
            encoder_block="residual",
            decoder_block="residual",
            use_attention_gates=True,
        )
        model.eval()
        x = torch.randn(1, 1, 64, 64)
        out = model(x)
        assert out.shape == (1, 1, 64, 64)

    def test_residual_with_se(self):
        """Residual + SE should work together."""
        model = UNet(
            encoder_block="residual",
            decoder_block="residual",
            use_se=True,
        )
        model.eval()
        x = torch.randn(1, 1, 64, 64)
        out = model(x)
        assert out.shape == (1, 1, 64, 64)

    def test_residual_with_cbam(self):
        """Residual + CBAM should work together."""
        model = UNet(
            encoder_block="residual",
            decoder_block="residual",
            use_cbam=True,
        )
        model.eval()
        x = torch.randn(1, 1, 64, 64)
        out = model(x)
        assert out.shape == (1, 1, 64, 64)

    def test_attention_with_se(self):
        """Attention gates + SE should work together."""
        model = UNet(
            use_attention_gates=True,
            use_se=True,
        )
        model.eval()
        x = torch.randn(1, 1, 64, 64)
        out = model(x)
        assert out.shape == (1, 1, 64, 64)

    def test_full_configuration(self):
        """All modules together (except CBAM) should work."""
        model = UNet(
            encoder_block="residual",
            decoder_block="residual",
            bridge_type="aspp",
            use_attention_gates=True,
            use_se=True,
            use_deep_supervision=True,
        )
        model.eval()
        x = torch.randn(1, 1, 256, 256)
        out = model(x)
        assert out.shape == (1, 1, 256, 256)

    def test_lightweight_config(self):
        """Lightweight UNet with reduced features should work."""
        model = UNet(features=[32, 64, 128, 256])
        model.eval()
        x = torch.randn(1, 1, 64, 64)
        out = model(x)
        assert out.shape == (1, 1, 64, 64)

    def test_bilinear_upsampling(self):
        """Bilinear upsampling mode should work."""
        model = UNet(upsample_mode="bilinear")
        model.eval()
        x = torch.randn(1, 1, 64, 64)
        out = model(x)
        assert out.shape == (1, 1, 64, 64)

    def test_deep_unet(self):
        """Deeper UNet with 5 levels should work."""
        model = UNet(features=[64, 128, 256, 512, 1024])
        model.eval()
        x = torch.randn(1, 1, 256, 256)
        out = model(x)
        assert out.shape == (1, 1, 256, 256)


class TestGradientFlow:
    """Test gradient flow through models."""

    def test_baseline_gradient_flow(self):
        """Gradients should flow through baseline UNet."""
        model = UNet(in_channels=1, out_channels=1)
        x = torch.randn(1, 1, 64, 64, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_deep_supervision_gradient_flow(self):
        """Gradients should flow through deep supervision outputs."""
        model = UNet(
            in_channels=1,
            out_channels=1,
            use_deep_supervision=True,
        )
        model.train()
        x = torch.randn(1, 1, 64, 64, requires_grad=True)
        main_out, aux_outs = model(x)

        # Loss from all outputs
        loss = main_out.sum()
        for aux in aux_outs:
            loss = loss + aux.sum()

        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestBackwardCompatibility:
    """Ensure ModularUNet alias works."""

    def test_modular_unet_alias(self):
        """ModularUNet should be an alias for UNet."""
        assert ModularUNet is UNet

    def test_modular_unet_instantiation(self):
        """ModularUNet should instantiate correctly."""
        model = ModularUNet(in_channels=1, out_channels=1)
        x = torch.randn(1, 1, 64, 64)
        out = model(x)
        assert out.shape == (1, 1, 64, 64)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
