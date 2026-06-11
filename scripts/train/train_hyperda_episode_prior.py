#!/usr/bin/env python3
"""Scaffold for deterministic prompt_to_zeta_prior training.

The planned model learns a deterministic prompt_to_zeta_prior from source
episode bank artifacts and selects hyperparameters with
source_side_episodic_validation. It does not use target_val, target_eval, or
target labels for prior training or hyperparameter selection.
"""
from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train deterministic HyperDA prompt_to_zeta_prior (scaffold)."
    )
    parser.add_argument("--episode_bank_dir", required=True)
    parser.add_argument("--output_dir", default="artifacts/hyperda_episode_prior")
    parser.add_argument("--validation_source", default="source_side_episodic_validation")
    return parser.parse_args()


def main() -> None:
    _ = parse_args()
    raise NotImplementedError(
        "HyperDA prompt_to_zeta_prior training is scaffolded only. Implement "
        "zeta schema loading, deterministic prompt-to-zeta regression, and "
        "source_side_episodic_validation before running this entrypoint; it "
        "does not use target_val or target_eval."
    )


if __name__ == "__main__":
    main()
