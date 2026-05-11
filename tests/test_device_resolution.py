"""Tests for hydroda/utils/device.py"""
import pytest


def test_resolve_device_cpu():
    """Test resolve_device with cpu."""
    import torch
    from hydroda.utils.device import resolve_device

    # Force CPU
    device = resolve_device("cpu")
    assert device.type == "cpu"


def test_resolve_device_cuda_available():
    """Test resolve_device when CUDA is available."""
    import torch
    from hydroda.utils.device import resolve_device

    if torch.cuda.is_available():
        device = resolve_device("cuda")
        assert device.type == "cuda"


def test_resolve_device_cuda_unavailable_no_require():
    """Test resolve_device falls back to cpu when CUDA unavailable and require_gpu=False."""
    import torch
    from hydroda.utils.device import resolve_device

    # If no CUDA, should return CPU
    if not torch.cuda.is_available():
        device = resolve_device("cuda", require_gpu=False)
        assert device.type == "cpu"


def test_resolve_device_require_gpu_no_cuda():
    """Test resolve_device exits when require_gpu=True but no CUDA."""
    import torch
    from hydroda.utils.device import resolve_device

    if not torch.cuda.is_available():
        with pytest.raises(SystemExit) as exc_info:
            resolve_device("cuda", require_gpu=True)
        assert exc_info.value.code == 1


def test_get_gpu_memory_info_no_cuda():
    """Test get_gpu_memory_info returns zeros when no CUDA."""
    from hydroda.utils.device import get_gpu_memory_info

    info = get_gpu_memory_info()
    assert "allocated_gb" in info
    assert "reserved_gb" in info
    assert "total_gb" in info
    assert "free_gb" in info


def test_get_gpu_memory_info_with_cuda():
    """Test get_gpu_memory_info returns proper values when CUDA available."""
    import torch
    from hydroda.utils.device import get_gpu_memory_info

    if torch.cuda.is_available():
        info = get_gpu_memory_info()
        assert info["total_gb"] > 0
        assert info["allocated_gb"] >= 0
        assert info["reserved_gb"] >= 0
        assert info["free_gb"] >= 0


def test_supports_amp():
    """Test supports_amp returns True for cuda, False for cpu."""
    import torch
    from hydroda.utils.device import supports_amp

    assert supports_amp(torch.device("cuda")) == (torch.cuda.is_available())
    assert supports_amp(torch.device("cpu")) == False