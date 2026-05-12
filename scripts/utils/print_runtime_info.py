#!/usr/bin/env python3
"""Lightweight runtime diagnostics script for HydroDA-OOD / HyperDA V4.

Usage:
    PYTHONPATH=. python scripts/utils/print_runtime_info.py

Lighter than check_environment.py — prints only runtime info needed
for debugging and logging.
"""

import socket
import sys

from hydroda.utils.runtime import get_git_hash


def main():
    print("=" * 50)
    print("HydroDA-OOD Runtime Info")
    print("=" * 50)

    # Python / torch
    try:
        import torch
        print(f"Python: {sys.version.split()[0]}")
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            if torch.cuda.device_count() > 0:
                print(f"GPU 0: {torch.cuda.get_device_name(0)}")
        else:
            print("Device: CPU only")
    except ImportError:
        print("PyTorch not installed")

    # Host / env
    print(f"Hostname: {socket.gethostname()}")
    import os
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '(not set)')}")
    print(f"CONDA_ENV: {os.environ.get('CONDA_DEFAULT_ENV', '(not set)')}")

    # Git hash
    print(f"Git: {get_git_hash()}")

    print("=" * 50)


if __name__ == "__main__":
    main()