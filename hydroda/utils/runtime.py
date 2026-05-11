"""Shared runtime utilities for HydroDA-OOD / HyperDA V4."""

from __future__ import annotations

import os
import socket
import subprocess
from typing import Optional


def get_git_hash() -> str:
    """Return 12-char git hash of current HEAD."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()[:12]
    except Exception:
        return "unknown"


def get_git_status() -> str:
    """Return short git status."""
    try:
        return subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def get_hostname() -> str:
    """Return hostname."""
    return socket.gethostname()


def get_conda_env_name() -> str:
    """Return conda environment name or 'unknown'."""
    return os.environ.get("CONDA_DEFAULT_ENV", "unknown")


def get_cuda_visible_devices() -> str:
    """Return CUDA_VISIBLE_DEVICES value or '' if not set."""
    return os.environ.get("CUDA_VISIBLE_DEVICES", "")


def get_timestamp() -> str:
    """Return UTC ISO timestamp."""
    from datetime import datetime as _dt
    return _dt.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")


def format_elapsed(seconds: float) -> str:
    """Format elapsed time as human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        return f"{seconds/3600:.1f}h"


def gather_runtime_info() -> dict:
    """Gather all runtime info as a dict for run artifact."""
    import torch
    info = {
        "timestamp": get_timestamp(),
        "hostname": get_hostname(),
        "conda_env": get_conda_env_name(),
        "git_hash": get_git_hash(),
        "git_status": get_git_status(),
        "cuda_visible_devices": get_cuda_visible_devices(),
        "python_version": __import__("sys").version,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["cudnn_enabled"] = torch.backends.cudnn.enabled
        info["gpu_count"] = torch.cuda.device_count()
        if torch.cuda.device_count() > 0:
            info["gpu_name"] = torch.cuda.get_device_name(0)
            gpu = torch.cuda
            allocated = gpu.memory_allocated(0) / 1e9
            reserved = gpu.memory_reserved(0) / 1e9
            total = gpu.get_device_properties(0).total_memory / 1e9
            info["gpu_allocated_gb"] = round(allocated, 2)
            info["gpu_reserved_gb"] = round(reserved, 2)
            info["gpu_total_gb"] = round(total, 2)
            info["gpu_free_gb"] = round(total - reserved, 2)
    return info