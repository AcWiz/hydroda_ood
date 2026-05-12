"""Unified time decoding utilities for HydroDA-OOD / HyperDA V4.

All functions operate in UTC to eliminate local timezone risks.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Union

import numpy as np


def decode_da_time(timestamp: Union[int, float, np.integer, np.floating]) -> datetime:
    """Decode a single DA.nc timestamp (Unix seconds) to UTC datetime.

    Args:
        timestamp: Unix timestamp in seconds.

    Returns:
        Timezone-aware UTC datetime.
    """
    return datetime.fromtimestamp(float(timestamp), tz=timezone.utc)


def decode_da_time_xr(time_array: np.ndarray) -> List[datetime]:
    """Decode an xarray time coordinate array to UTC datetime list.

    Args:
        time_array: 1-D numpy array of Unix timestamps (int64 or float64).

    Returns:
        List of timezone-aware UTC datetime objects.
    """
    return [datetime.fromtimestamp(float(t), tz=timezone.utc) for t in time_array.astype(np.float64)]


def year_from_timestamp(timestamp: Union[int, float, np.integer, np.floating]) -> int:
    """Extract year from a DA.nc timestamp (Unix seconds).

    Args:
        timestamp: Unix timestamp in seconds.

    Returns:
        Integer year (UTC).
    """
    return datetime.fromtimestamp(float(timestamp), tz=timezone.utc).year


def month_from_timestamp(timestamp: Union[int, float, np.integer, np.floating]) -> int:
    """Extract month (1-12) from a DA.nc timestamp.

    Args:
        timestamp: Unix timestamp in seconds.

    Returns:
        Integer month 1-12 (UTC).
    """
    return datetime.fromtimestamp(float(timestamp), tz=timezone.utc).month


def date_str_from_timestamp(timestamp: Union[int, float, np.integer, np.floating]) -> str:
    """Format a DA.nc timestamp as YYYY-MM-DD string.

    Args:
        timestamp: Unix timestamp in seconds.

    Returns:
        Date string in YYYY-MM-DD format (UTC).
    """
    dt = datetime.fromtimestamp(float(timestamp), tz=timezone.utc)
    return dt.strftime("%Y-%m-%d")
