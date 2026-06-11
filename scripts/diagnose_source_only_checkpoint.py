#!/usr/bin/env python3
"""Compatibility wrapper for the source-only diagnostic script.

The implementation lives in ``scripts/analysis/diagnose_source_only_checkpoint.py``.
This module keeps older tests and command lines that import from ``scripts/``
working without duplicating diagnostic logic.
"""
from __future__ import annotations

from analysis.diagnose_source_only_checkpoint import *  # noqa: F401,F403
from analysis.diagnose_source_only_checkpoint import main


if __name__ == "__main__":
    main()
