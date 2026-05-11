"""Device management utilities for HydroDA-OOD / HyperDA V4."""

from __future__ import annotations

import torch
from typing import Optional


def resolve_device(device_arg: str = "cuda", require_gpu: bool = False) -> torch.device:
    """Resolve and validate device.

    Args:
        device_arg: Device string ('cuda', 'cpu', 'cuda:0', etc.)
        require_gpu: If True, exit with error if CUDA unavailable

    Returns:
        torch.device

    Raises:
        SystemExit: If require_gpu=True and no CUDA device available
    """
    if device_arg == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            if require_gpu:
                print("ERROR: --require_gpu set but no CUDA device available.")
                raise SystemExit(1)
            return torch.device("cpu")
    elif device_arg.startswith("cuda:"):
        if torch.cuda.is_available():
            idx = int(device_arg.split(":")[1])
            if idx >= torch.cuda.device_count():
                print(f"WARNING: cuda:{idx} not available (max {torch.cuda.device_count()-1}), using cpu")
                return torch.device("cpu")
            return torch.device(device_arg)
        else:
            if require_gpu:
                print("ERROR: --require_gpu set but no CUDA device available.")
                raise SystemExit(1)
            return torch.device("cpu")
    else:
        return torch.device(device_arg)


def get_gpu_memory_info() -> dict:
    """Get GPU memory info for device 0.

    Returns:
        dict with allocated_gb, reserved_gb, total_gb, free_gb
    """
    if not torch.cuda.is_available():
        return {"allocated_gb": 0.0, "reserved_gb": 0.0, "total_gb": 0.0, "free_gb": 0.0}

    allocated = torch.cuda.memory_allocated(0) / 1e9
    reserved = torch.cuda.memory_reserved(0) / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    return {
        "allocated_gb": round(allocated, 2),
        "reserved_gb": round(reserved, 2),
        "total_gb": round(total, 2),
        "free_gb": round(total - reserved, 2),
    }


def log_device_summary() -> None:
    """Print GPU info to console."""
    print("=" * 50)
    print("Device Summary")
    print("=" * 50)
    print(f"  torch:        {torch.__version__}")
    print(f"  cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  cuda version:  {torch.version.cuda}")
        print(f"  cudnn enabled: {torch.backends.cudnn.enabled}")
        print(f"  gpu count:    {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mem_total = props.total_memory / 1e9
            mem_alloc = torch.cuda.memory_allocated(i) / 1e9
            mem_res = torch.cuda.memory_reserved(i) / 1e9
            print(f"  GPU {i}: {props.name}")
            print(f"    memory: total={mem_total:.1f}GB allocated={mem_alloc:.1f}GB reserved={mem_res:.1f}GB")
    else:
        print("  No GPU — running on CPU")
    print("=" * 50)


def supports_amp(device: torch.device) -> bool:
    """Check if device supports automatic mixed precision."""
    return device.type == "cuda"