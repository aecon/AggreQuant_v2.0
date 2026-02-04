"""Unit tests for neural network architectures.

Original author: Athena Economides
Refactoring tool: Claude Opus 4.5
Date: 2026-02-04
"""

import pytest
import torch
import torch.nn as nn

from aggrequant.nn.architectures import (
    ModularUNet,
    create_model,
    list_architectures,
    BENCHMARK_CONFIGS,
    get_config,
)
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


class TestModularUNet:
    """Test ModularUNet architecture."""

    def test_baseline_forward(self):
        """Baseline UNet should produce correct output shape."""
        model = ModularUNet(in_channels=1, out_channels=1)
        x = torch.randn(2, 1, 256, 256)
        out = model(x)
        assert out.shape == (2, 1, 256, 256)

    def test_residual_forward(self):
        """Residual UNet should produce correct output shape."""
        model = ModularUNet(
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
        model = ModularUNet(
            in_channels=1,
            out_channels=1,
            use_attention_gates=True,
        )
        x = torch.randn(2, 1, 256, 256)
        out = model(x)
        assert out.shape == (2, 1, 256, 256)

    def test_se_forward(self):
        """UNet with SE blocks should produce correct output shape."""
        model = ModularUNet(
            in_channels=1,
            out_channels=1,
            use_se=True,
        )
        x = torch.randn(2, 1, 256, 256)
        out = model(x)
        assert out.shape == (2, 1, 256, 256)

    def test_cbam_forward(self):
        """UNet with CBAM should produce correct output shape."""
        model = ModularUNet(
            in_channels=1,
            out_channels=1,
            use_cbam=True,
        )
        x = torch.randn(2, 1, 256, 256)
        out = model(x)
        assert out.shape == (2, 1, 256, 256)

    def test_se_cbam_mutually_exclusive(self):
        """SE and CBAM should be mutually exclusive."""
        with pytest.raises(ValueError):
            ModularUNet(use_se=True, use_cbam=True)

    def test_deep_supervision_training(self):
        """Deep supervision should return tuple during training."""
        model = ModularUNet(
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
        model = ModularUNet(
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
        model = ModularUNet(
            in_channels=1,
            out_channels=1,
            features=[32, 64, 128, 256],
        )
        x = torch.randn(2, 1, 256, 256)
        out = model(x)
        assert out.shape == (2, 1, 256, 256)

    def test_multichannel_input(self):
        """UNet should handle multi-channel input."""
        model = ModularUNet(in_channels=3, out_channels=1)
        x = torch.randn(2, 3, 256, 256)
        out = model(x)
        assert out.shape == (2, 1, 256, 256)

    def test_multichannel_output(self):
        """UNet should handle multi-channel output."""
        model = ModularUNet(in_channels=1, out_channels=3)
        x = torch.randn(2, 1, 256, 256)
        out = model(x)
        assert out.shape == (2, 3, 256, 256)

    def test_non_power_of_two_input(self):
        """UNet should handle non-power-of-two input sizes."""
        model = ModularUNet(in_channels=1, out_channels=1)
        x = torch.randn(2, 1, 200, 200)
        out = model(x)
        assert out.shape == (2, 1, 200, 200)

    def test_count_parameters(self):
        """count_parameters should return positive integer."""
        model = ModularUNet()
        params = model.count_parameters()
        assert params > 0
        assert isinstance(params, int)


class TestModelFactory:
    """Test model factory functions."""

    def test_list_architectures(self):
        """list_architectures should return non-empty list."""
        archs = list_architectures()
        assert isinstance(archs, list)
        assert len(archs) > 0
        assert "unet_baseline" in archs

    def test_create_model_baseline(self):
        """create_model should create baseline UNet."""
        model = create_model("unet_baseline", in_channels=1, out_channels=1)
        assert isinstance(model, ModularUNet)

    def test_create_model_residual(self):
        """create_model should create residual UNet."""
        model = create_model("unet_residual", in_channels=1, out_channels=1)
        assert isinstance(model, ModularUNet)

    def test_create_model_unknown(self):
        """create_model should raise error for unknown architecture."""
        with pytest.raises(ValueError):
            create_model("unknown_architecture")

    def test_create_model_overrides(self):
        """create_model should accept override kwargs."""
        model = create_model(
            "unet_baseline",
            in_channels=3,
            out_channels=2,
            features=[32, 64, 128, 256],
        )
        x = torch.randn(1, 3, 128, 128)
        out = model(x)
        assert out.shape == (1, 2, 128, 128)


class TestBenchmarkConfigs:
    """Test benchmark configurations."""

    def test_benchmark_configs_not_empty(self):
        """BENCHMARK_CONFIGS should contain configurations."""
        assert len(BENCHMARK_CONFIGS) > 0

    def test_get_config(self):
        """get_config should return configuration dict."""
        config = get_config("unet_baseline")
        assert isinstance(config, dict)
        assert "encoder_block" in config

    def test_get_config_unknown(self):
        """get_config should raise error for unknown config."""
        with pytest.raises(KeyError):
            get_config("unknown_config")

    def test_all_configs_create_valid_models(self):
        """All benchmark configs should create valid models."""
        # Configs that require larger input sizes due to ASPP or 5 pooling levels
        large_configs = {"unet_aspp", "unet_deep", "unet_deep_res_attention"}

        for name in BENCHMARK_CONFIGS:
            model = create_model(name, in_channels=1, out_channels=1)
            model.eval()  # Use eval mode to avoid BatchNorm issues with small batches

            if name in large_configs:
                # These need larger input due to ASPP dilations or 5 pooling levels
                x = torch.randn(1, 1, 256, 256)
                out = model(x)
                if isinstance(out, tuple):
                    out = out[0]
                assert out.shape == (1, 1, 256, 256), f"Failed for {name}"
            else:
                x = torch.randn(1, 1, 64, 64)
                out = model(x)
                # Handle deep supervision
                if isinstance(out, tuple):
                    out = out[0]
                assert out.shape == (1, 1, 64, 64), f"Failed for {name}"


class TestGradientFlow:
    """Test gradient flow through models."""

    def test_baseline_gradient_flow(self):
        """Gradients should flow through baseline UNet."""
        model = ModularUNet(in_channels=1, out_channels=1)
        x = torch.randn(1, 1, 64, 64, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_deep_supervision_gradient_flow(self):
        """Gradients should flow through deep supervision outputs."""
        model = ModularUNet(
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
