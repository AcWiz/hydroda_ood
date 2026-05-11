"""K-Date Selection Utilities for Leave-One-Region-Out Splits.

No-leakage declaration:
    Support dates are selected ONLY via:
    - Calendar constraints (quarter/month/half-month rules)
    - Time availability in 2022
    - base_valid_mask coverage threshold

    NOT via:
    - Analysis increment values
    - Model errors
    - Target query label distribution
    - Future query statistics
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable, Dict, List, Tuple

import numpy as np


# Quarter definitions: (month_start, month_end) inclusive
QUARTERS: Dict[int, Tuple[int, int]] = {
    1: (1, 3),   # DJF - Winter
    2: (4, 6),   # MAM - Spring
    3: (7, 9),   # JJA - Summer
    4: (10, 12), # SON - Fall
}

# Month halves: days 1-15 (first half), days 16-31 (second half)
HALF_MONTHS: Dict[int, List[int]] = {
    1: list(range(1, 16)),
    2: list(range(16, 32)),
}


def _select_by_bucket(
    available_dates: List[Tuple[int, datetime]],
    valid_mask: np.ndarray,
    seed: int,
    bucket_filter: Callable[[datetime], bool],
) -> List[Tuple[int, datetime]]:
    """Select one valid date per time bucket.

    Args:
        available_dates: List of (time_index, datetime) candidates
        valid_mask: Boolean array indicating which dates are valid
        seed: Random seed for deterministic selection
        bucket_filter: Function that returns True if datetime is in desired bucket

    Returns:
        List of selected (time_index, datetime) tuples (one per bucket)
    """
    rng = np.random.RandomState(seed)
    bucket_dates = [
        (idx, dt) for (idx, dt), v in zip(available_dates, valid_mask)
        if v and bucket_filter(dt)
    ]
    if not bucket_dates:
        return []
    chosen = bucket_dates[rng.randint(len(bucket_dates))]
    return [chosen]


def select_support_dates_k0(
    available_dates: List[Tuple[int, datetime]],
    seed: int,
) -> List[Tuple[int, datetime]]:
    """K=0: Zero support dates. No adaptation from target support."""
    return []


def select_support_dates_k4(
    available_dates: List[Tuple[int, datetime]],
    valid_mask: np.ndarray,
    seed: int,
) -> List[Tuple[int, datetime]]:
    """K=4: Select one valid date per quarter (one per season)."""
    rng = np.random.RandomState(seed)
    selected = []
    for q in [1, 2, 3, 4]:
        month_start, month_end = QUARTERS[q]
        bucket_dates = [
            (idx, dt) for (idx, dt), v in zip(available_dates, valid_mask)
            if v and month_start <= dt.month <= month_end
        ]
        if not bucket_dates:
            continue
        chosen = bucket_dates[rng.randint(len(bucket_dates))]
        selected.append(chosen)
    return selected


def select_support_dates_k12(
    available_dates: List[Tuple[int, datetime]],
    valid_mask: np.ndarray,
    seed: int,
) -> List[Tuple[int, datetime]]:
    """K=12: Select one valid date per month."""
    rng = np.random.RandomState(seed)
    selected = []
    for m in range(1, 13):
        bucket_dates = [
            (idx, dt) for (idx, dt), v in zip(available_dates, valid_mask)
            if v and dt.month == m
        ]
        if not bucket_dates:
            continue
        chosen = bucket_dates[rng.randint(len(bucket_dates))]
        selected.append(chosen)
    return selected


def select_support_dates_k24(
    available_dates: List[Tuple[int, datetime]],
    valid_mask: np.ndarray,
    seed: int,
) -> List[Tuple[int, datetime]]:
    """K=24: Select one valid date per half-month."""
    rng = np.random.RandomState(seed)
    selected = []
    for m in range(1, 13):
        for half in [1, 2]:
            days = HALF_MONTHS[half]
            bucket_dates = [
                (idx, dt) for (idx, dt), v in zip(available_dates, valid_mask)
                if v and dt.month == m and dt.day in days
            ]
            if not bucket_dates:
                continue
            chosen = bucket_dates[rng.randint(len(bucket_dates))]
            selected.append(chosen)
    return selected


# K -> (selector_func, needed_args) dispatch table
_K_SELECTORS: Dict[int, Tuple[Callable, str]] = {
    0: (select_support_dates_k0, "seed"),
    4: (select_support_dates_k4, "valid_mask, seed"),
    12: (select_support_dates_k12, "valid_mask, seed"),
    24: (select_support_dates_k24, "valid_mask, seed"),
}


def get_support_dates_for_K(
    available_dates: List[Tuple[int, datetime]],
    valid_mask: np.ndarray,
    K: int,
    seed: int,
) -> List[Tuple[int, datetime]]:
    """Dispatch to appropriate K-date selection function.

    Args:
        available_dates: List of (time_index, datetime) in support year
        valid_mask: Boolean array of which dates are valid
        K: Number of support dates for main experiments (0, 4, or 12)
        seed: Random seed

    Returns:
        List of selected (time_index, datetime) tuples
    """
    if K not in _K_SELECTORS:
        raise ValueError(f"Unsupported K={K}. Must be one of {{0, 4, 12, 24}}")
    selector = _K_SELECTORS[K][0]
    if K == 0:
        return selector(available_dates, seed)
    return selector(available_dates, valid_mask, seed)


def dates_to_serializable(dates: List[Tuple[int, datetime]]) -> List[dict]:
    """Convert datetime list to JSON-serializable format.

    Args:
        dates: List of (time_index, datetime) tuples

    Returns:
        List of dicts with 'time_index', 'date_str' (YYYY-MM-DD), 'datetime_str' (ISO)
    """
    return [
        {
            "time_index": int(idx),
            "date_str": dt.strftime("%Y-%m-%d"),
            "datetime_str": dt.isoformat(),
        }
        for idx, dt in dates
    ]