#!/usr/bin/env python3
"""Canonical builder for V4.2 target_full_train split manifests.

This module delegates to `build_kdate_splits.py`, which now contains the
shared implementation for both the main full-target-train protocol and legacy
few-shot K-date ablations. New workflows should call this script.
"""

from __future__ import annotations

from build_kdate_splits import main


if __name__ == "__main__":
    main()
