"""Tests for UNet architecture and model registry."""

import torch
import pytest

from aggrequant.nn.architectures.unet import UNet, _make_conv_block, _make_channel_attention
from aggrequant.nn.architectures.registry import create_model, list_models, describe_models


SMALL_FEATURES = [8, 16, 32, 64]
INPUT_SHAPE = (1, 1, 64, 64)


# --- Fixtures ---

@pytest.fixture(scope="module")
def x():
    return torch.randn(*INPUT_SHAPE)


# --- _make_conv_block helper ---

@pytest.mark.parametrize("block_type", ["double_conv", "residual", "convnext"])
def test_make_conv_block_output_shape(block_type):
    block = _make_conv_block(32, 64, block_type)
    out = block(torch.randn(1, 32, 32, 32))
    assert out.shape == (1, 64, 32, 32)


# --- _make_channel_attention helper ---

def test_make_channel_attention_none():
    assert _make_channel_attention(32) is None

@pytest.mark.parametrize("kwargs", [
    {"use_se": True},
    {"use_cbam": True},
    {"use_eca": True},
])
def test_make_channel_attention_output_shape(kwargs):
    attn = _make_channel_attention(32, **kwargs)
    assert attn is not None
    out = attn(torch.randn(1, 32, 16, 16))
    assert out.shape == (1, 32, 16, 16)


# --- UNet forward pass ---

def test_unet_baseline_output_shape(x):
    model = UNet(features=SMALL_FEATURES)
    out = model(x)
    assert out.shape == INPUT_SHAPE

@pytest.mark.parametrize("encoder_type", ["double_conv", "residual", "convnext"])
def test_unet_encoder_variants(encoder_type, x):
    model = UNet(encoder_block=encoder_type, features=SMALL_FEATURES)
    out = model(x)
    assert out.shape == INPUT_SHAPE

@pytest.mark.parametrize("bridge_type", ["double_conv", "residual", "aspp"])
def test_unet_bridge_variants(bridge_type, x):
    model = UNet(bridge_type=bridge_type, features=SMALL_FEATURES)
    out = model(x)
    assert out.shape == INPUT_SHAPE

def test_unet_attention_gates(x):
    model = UNet(use_attention_gates=True, features=SMALL_FEATURES)
    out = model(x)
    assert out.shape == INPUT_SHAPE

@pytest.mark.parametrize("attn_kwarg", [
    {"use_se": True},
    {"use_cbam": True},
    {"use_eca": True},
])
def test_unet_channel_attention_variants(attn_kwarg, x):
    model = UNet(**attn_kwarg, features=SMALL_FEATURES)
    out = model(x)
    assert out.shape == INPUT_SHAPE

def test_unet_mutually_exclusive_attention():
    with pytest.raises(ValueError, match="mutually exclusive"):
        UNet(use_se=True, use_cbam=True, features=SMALL_FEATURES)

def test_unet_custom_in_out_channels():
    model = UNet(in_channels=3, out_channels=2, features=SMALL_FEATURES)
    out = model(torch.randn(1, 3, 64, 64))
    assert out.shape == (1, 2, 64, 64)

def test_unet_bilinear_upsample(x):
    model = UNet(upsample_mode="bilinear", features=SMALL_FEATURES)
    out = model(x)
    assert out.shape == INPUT_SHAPE


# --- Deep supervision ---

def test_unet_deep_supervision_training():
    model = UNet(use_deep_supervision=True, features=SMALL_FEATURES)
    model.train()
    out = model(torch.randn(*INPUT_SHAPE))
    assert isinstance(out, tuple)
    main, aux = out
    assert main.shape == INPUT_SHAPE
    assert len(aux) == len(SMALL_FEATURES) - 1

def test_unet_deep_supervision_eval(x):
    model = UNet(use_deep_supervision=True, features=SMALL_FEATURES)
    model.eval()
    out = model(x)
    assert isinstance(out, torch.Tensor)
    assert out.shape == INPUT_SHAPE


# --- UNet utilities ---

def test_unet_count_parameters():
    model = UNet(features=SMALL_FEATURES)
    assert model.count_parameters() > 0

def test_unet_get_config():
    model = UNet(use_deep_supervision=True, use_attention_gates=True, features=SMALL_FEATURES)
    config = model.get_config()
    assert config["use_deep_supervision"] is True
    assert config["use_attention_gates"] is True
    assert config["features"] == SMALL_FEATURES

def test_unet_repr():
    model = UNet(features=SMALL_FEATURES)
    r = repr(model)
    assert "ModularUNet" in r
    assert "params=" in r


# --- Gradient flow ---

def test_unet_gradient_flows():
    model = UNet(features=SMALL_FEATURES)
    x = torch.randn(*INPUT_SHAPE, requires_grad=True)
    out = model(x)
    out.sum().backward()
    assert x.grad is not None
    assert torch.all(torch.isfinite(x.grad))


# --- Registry: list_models / describe_models ---

EXPECTED_MODELS = [
    "baseline", "resunet", "attention_resunet", "se_attention_resunet",
    "aspp_se_attention_resunet", "convnext_unet", "eca_attention_resunet",
]

def test_list_models_returns_all_presets():
    names = list_models()
    for expected in EXPECTED_MODELS:
        assert expected in names

def test_describe_models_returns_dict():
    descriptions = describe_models()
    assert isinstance(descriptions, dict)
    for name in EXPECTED_MODELS:
        assert name in descriptions
        assert isinstance(descriptions[name], str)
        assert len(descriptions[name]) > 0


# --- Registry: create_model ---

@pytest.mark.parametrize("name", EXPECTED_MODELS)
def test_create_model_forward_pass(name):
    model = create_model(name, features=SMALL_FEATURES)
    out = model(torch.randn(*INPUT_SHAPE))
    assert out.shape == INPUT_SHAPE

def test_create_model_unknown_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        create_model("nonexistent")

def test_create_model_override_kwargs():
    model = create_model("baseline", in_channels=3, features=SMALL_FEATURES)
    out = model(torch.randn(1, 3, 64, 64))
    assert out.shape == INPUT_SHAPE
