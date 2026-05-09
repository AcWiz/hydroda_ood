"""
File hashing utilities for HydroDA-OOD artifact verification.

Provides SHA256 checksum computation for artifact files.
"""

import hashlib
from pathlib import Path


def compute_sha256(filepath: Path | str) -> str:
    """
    Compute SHA256 checksum of a file by reading in chunks.

    Args:
        filepath: Path to the file.

    Returns:
        Hexadecimal SHA256 string.
    """
    sha256 = hashlib.sha256()
    path = Path(filepath)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()