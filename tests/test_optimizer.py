"""tests/test_optimizer.py - Unit tests for Quantora's optimizer module."""

import pytest
import torch
from optimizer import (
    get_hardware_capabilities,
    validate_configuration,
    get_model_loading_kwargs,
    calculate_kv_cache_size_bytes,
    format_bytes_to_mb,
)


def test_hardware_detection():
    """Verify hardware detection returns all expected keys and valid types."""
    hw = get_hardware_capabilities()
    assert isinstance(hw, dict)
    expected_keys = [
        "os", "python_version", "torch_version", "cuda_available",
        "device_name", "total_vram_gb", "compute_capability",
        "bf16_supported", "bitsandbytes_available", "peft_available",
        "total_ram_gb", "available_ram_gb"
    ]
    for key in expected_keys:
        assert key in hw, f"Missing key: {key}"
    assert isinstance(hw["cuda_available"], bool)
    assert isinstance(hw["total_ram_gb"], (int, float))


def test_validate_configuration_cpu():
    """Verify configuration validation on CPU device."""
    # FP32 is always valid on CPU
    valid, reason = validate_configuration("fp32", "cpu")
    assert valid is True
    assert "valid" in reason.lower() or "supported" in reason.lower()

    # INT8 and INT4 must be rejected on CPU
    valid, reason = validate_configuration("int8", "cpu")
    assert valid is False
    assert "gpu" in reason.lower() or "cuda" in reason.lower()

    valid, reason = validate_configuration("int4", "cpu")
    assert valid is False
    assert "gpu" in reason.lower() or "cuda" in reason.lower()

    # Invalid precision name
    valid, reason = validate_configuration("invalid_prec", "cpu")
    assert valid is False


def test_get_model_loading_kwargs():
    """Verify model loading arguments generation for FP32 and FP16."""
    kwargs_fp32 = get_model_loading_kwargs("fp32", "cpu")
    assert kwargs_fp32.get("torch_dtype") == torch.float32
    assert kwargs_fp32.get("device_map") == "cpu"

    if torch.cuda.is_available():
        kwargs_fp16 = get_model_loading_kwargs("fp16", "cuda")
        assert kwargs_fp16.get("torch_dtype") == torch.float16
        assert kwargs_fp16.get("device_map") == "cuda:0"


def test_kv_cache_calculations():
    """Verify theoretical KV cache memory footprint arithmetic."""
    # 24 layers, 16 heads, 64 dim, 1024 seq_length, batch_size=1, 2 bytes (FP16)
    # Size = 2 * 24 * 16 * 64 * 1024 * 1 * 2 = 100,663,296 bytes
    expected_bytes = 2 * 24 * 16 * 64 * 1024 * 1 * 2
    calculated_bytes = calculate_kv_cache_size_bytes(
        num_layers=24,
        num_heads=16,
        head_dim=64,
        seq_length=1024,
        batch_size=1,
        bytes_per_elem=2,
    )
    assert calculated_bytes == expected_bytes
    assert calculated_bytes == 100663296

    mb = format_bytes_to_mb(calculated_bytes)
    assert mb == 96.0  # 100663296 / (1024^2) = 96.0 MB


def test_quantize_model_cpu():
    """Verify PyTorch dynamic INT8 linear quantization for CPU."""
    from optimizer import quantize_model_cpu
    linear = torch.nn.Linear(16, 16)
    q_linear = quantize_model_cpu(linear)
    assert q_linear is not None
    x = torch.randn(1, 16)
    out = q_linear(x)
    assert out.shape == (1, 16)

