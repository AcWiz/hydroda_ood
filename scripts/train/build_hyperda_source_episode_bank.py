#!/usr/bin/env python3
"""Scaffold for the HyperDA source operator episode bank.

This entrypoint is intentionally protocol-safe while the deterministic GPD-style
prior is being built. It documents the source operator episode bank contract and
refuses to run experiments until the bank trainer, zeta schema, and pack/unpack
tests are implemented.

No-leakage declaration:
    - Episodes come from source_fit/source_val source regions only.
    - It must not read target_val or target_eval.
    - It must optimize target modules only: prompt residuals, adapter
      coefficient residuals, and monthly gain.
"""
from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build HyperDA source operator episode bank (scaffold)."
    )
    parser.add_argument("--source_checkpoint", required=True)
    parser.add_argument("--output_dir", default="artifacts/operator_bank/hyperda_source_episodes")
    parser.add_argument("--source_split", default="source_fit")
    parser.add_argument("--episode_axes", default="region,season,tile_group")
    return parser.parse_args()


def main() -> None:
    _ = parse_args()
    raise NotImplementedError(
        "HyperDA source operator episode bank is scaffolded only. Implement "
        "source_fit episode construction, target modules only optimization, "
        "and zeta pack/unpack tests before running this entrypoint."
    )


if __name__ == "__main__":
    main()
