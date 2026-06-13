#!/usr/bin/env python3
"""Compatibility entrypoint for the HyperDA source operator episode bank.

The real P4 v1 builder is ``build_source_episode_adapter_bank.py``. It creates
source-side adapter coefficient artifacts only; it does not train a generator,
use diffusion, or emit full-network parameters.

No-leakage declaration:
    - Episodes come from source_fit/source_val source regions only.
    - It must not read target_val or target_eval.
    - It must optimize target modules only: lightweight adapter coefficients.
"""
from __future__ import annotations

from scripts.train.build_source_episode_adapter_bank import main, parse_args


if __name__ == "__main__":
    main()
