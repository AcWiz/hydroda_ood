"""Protocol objects for HydroDA-OOD / HyperDA V4."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Iterable, List, Sequence


def _to_date(x: str | date | datetime) -> date:
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, date):
        return x
    return datetime.strptime(x[:10], "%Y-%m-%d").date()


@dataclass(frozen=True)
class DateRange:
    name: str
    start: date
    end: date

    @classmethod
    def from_strings(cls, name: str, start: str, end: str) -> "DateRange":
        return cls(name=name, start=_to_date(start), end=_to_date(end))

    def contains(self, value: str | date | datetime) -> bool:
        d = _to_date(value)
        return self.start <= d <= self.end


@dataclass(frozen=True)
class ProtocolConfig:
    """Single source of truth for V4 time and K-cycle protocol."""

    protocol_name: str = "HydroDA-OOD-HyperDA-V4"
    protocol_freeze_id: str = "hyperda_v4_2015_2025_k0_4_12"
    source_train: DateRange = field(
        default_factory=lambda: DateRange.from_strings("source_train", "2015-01-01", "2020-12-31")
    )
    source_fit: DateRange = field(
        default_factory=lambda: DateRange.from_strings("source_fit", "2015-01-01", "2019-12-31")
    )
    source_val: DateRange = field(
        default_factory=lambda: DateRange.from_strings("source_val", "2020-01-01", "2020-12-31")
    )
    target_context: DateRange = field(
        default_factory=lambda: DateRange.from_strings("target_context", "2021-01-01", "2021-12-31")
    )
    target_query: DateRange = field(
        default_factory=lambda: DateRange.from_strings("target_query", "2022-01-01", "2025-12-31")
    )
    main_K_values: Sequence[int] = (0, 4, 12)

    def role_for_date(self, value: str | date | datetime) -> str:
        if self.source_fit.contains(value):
            return "source_fit"
        if self.source_val.contains(value):
            return "source_val"
        if self.target_context.contains(value):
            return "target_context"
        if self.target_query.contains(value):
            return "target_query"
        return "outside_protocol"

    def assert_supported_K(self, K: int) -> None:
        if int(K) not in set(self.main_K_values):
            raise ValueError(f"Unsupported main K={K}. HyperDA V4 main experiments use {list(self.main_K_values)}.")

    def assert_no_query_dates(self, dates: Iterable[str | date | datetime], purpose: str) -> None:
        bad = [str(d) for d in dates if self.target_query.contains(d)]
        if bad:
            raise ValueError(f"Leakage risk: target_query dates used for {purpose}: {bad[:5]}")

    def assert_dates_within(self, dates: Iterable[str | date | datetime], allowed_roles: Sequence[str], purpose: str) -> None:
        allowed = set(allowed_roles)
        bad: List[str] = []
        for d in dates:
            role = self.role_for_date(d)
            if role not in allowed:
                bad.append(f"{d}:{role}")
        if bad:
            raise ValueError(f"Dates outside allowed roles for {purpose}: {bad[:8]}; allowed={sorted(allowed)}")
