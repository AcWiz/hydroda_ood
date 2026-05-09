"""HydroDA-OOD baselines."""

from hydroda.baselines.forecast import ForecastBaseline
from hydroda.baselines.mean_increment import (
    SourceMeanIncrementBaseline,
    TargetSupportMeanIncrementBaseline,
)
from hydroda.baselines.monthly_mean import TargetMonthlySupportIncrementBaseline
from hydroda.baselines.ridge import RidgeBaseline

__all__ = [
    "ForecastBaseline",
    "SourceMeanIncrementBaseline",
    "TargetSupportMeanIncrementBaseline",
    "TargetMonthlySupportIncrementBaseline",
    "RidgeBaseline",
]
