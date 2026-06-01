"""Protocol objects for HydroDA-OOD / HyperDA V4 historical target adaptation."""
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
    """Single source of truth for the current HydroDA-OOD time protocol.

    The main protocol is not K-shot/few-cycle adaptation. After source-domain
    hypernetwork training, the held-out target domain may use its full
    historical training period (2015-2021) for fast target adaptation. The
    target validation year is 2022, and final target evaluation is 2023-2025.
    Date ranges overlap between source_fit and target_train by design; callers
    must pass the intended role to assert_dates_within for domain-specific
    checks.
    """

    protocol_name: str = "HydroDA-OOD-HyperDA-V4"
    protocol_freeze_id: str = "hyperda_v4_3_historical_target_adapt_2015_2025_train2015_2021_val2022_test2023_2025"
    source_train: DateRange = field(
        default_factory=lambda: DateRange.from_strings("source_train", "2015-01-01", "2021-12-31")
    )
    source_fit: DateRange = field(
        default_factory=lambda: DateRange.from_strings("source_fit", "2015-01-01", "2021-12-31")
    )
    source_val: DateRange = field(
        default_factory=lambda: DateRange.from_strings("source_val", "2022-01-01", "2022-12-31")
    )
    source_test: DateRange = field(
        default_factory=lambda: DateRange.from_strings("source_test", "2023-01-01", "2025-12-31")
    )
    target_train: DateRange = field(
        default_factory=lambda: DateRange.from_strings("target_train", "2015-01-01", "2021-12-31")
    )
    target_adapt: DateRange = field(
        default_factory=lambda: DateRange.from_strings("target_adapt", "2015-01-01", "2021-12-31")
    )
    target_val: DateRange = field(
        default_factory=lambda: DateRange.from_strings("target_val", "2022-01-01", "2022-12-31")
    )
    # Backward-compatible date-range alias for older V4 docs/tests.
    target_context: DateRange = field(
        default_factory=lambda: DateRange.from_strings("target_context", "2015-01-01", "2021-12-31")
    )
    target_eval: DateRange = field(
        default_factory=lambda: DateRange.from_strings("target_eval", "2023-01-01", "2025-12-31")
    )
    # Backward-compatible date-range alias for older V4 docs/tests. The role
    # returned by role_for_date is target_eval.
    target_query: DateRange = field(
        default_factory=lambda: DateRange.from_strings("target_query", "2023-01-01", "2025-12-31")
    )
    main_adaptation_settings: Sequence[str] = ("target_full_train",)
    legacy_few_shot_K_values: Sequence[int] = (0, 4, 12)
    # Deprecated alias. Empty by design: K is no longer a main experiment axis.
    main_K_values: Sequence[int] = ()

    def role_for_date(self, value: str | date | datetime) -> str:
        if self.source_fit.contains(value):
            return "source_fit"
        if self.source_val.contains(value):
            return "source_val"
        if self.target_eval.contains(value):
            return "target_eval"
        return "outside_protocol"

    def assert_supported_adaptation_setting(self, setting: str) -> None:
        if setting not in set(self.main_adaptation_settings):
            raise ValueError(
                f"Unsupported main adaptation_setting={setting!r}. "
                f"Main protocol uses {list(self.main_adaptation_settings)}; "
                "few-shot K settings are legacy ablations."
            )

    def assert_legacy_few_shot_K(self, K: int) -> None:
        if int(K) not in set(self.legacy_few_shot_K_values):
            raise ValueError(
                f"Unsupported legacy few-shot K={K}. "
                f"Allowed legacy K values: {list(self.legacy_few_shot_K_values)}."
            )

    def assert_supported_K(self, K: int, *, allow_legacy: bool = False) -> None:
        """Deprecated K-axis validator.

        Main experiments must use adaptation_setting. Pass allow_legacy=True
        only for explicitly labeled few-shot ablation code paths.
        """
        if allow_legacy:
            self.assert_legacy_few_shot_K(K)
            return
        raise ValueError(
            "K is no longer a main protocol axis. Use adaptation_setting='target_full_train'. "
            "Pass allow_legacy=True only for secondary few-shot ablations."
        )

    def assert_no_query_dates(self, dates: Iterable[str | date | datetime], purpose: str) -> None:
        bad = [str(d) for d in dates if self.target_eval.contains(d)]
        if bad:
            raise ValueError(f"Leakage risk: target_eval dates used for {purpose}: {bad[:5]}")

    def assert_dates_within(self, dates: Iterable[str | date | datetime], allowed_roles: Sequence[str], purpose: str) -> None:
        allowed = self._expand_role_aliases(allowed_roles)
        bad: List[str] = []
        for d in dates:
            if not self._date_in_any_role(d, allowed):
                bad.append(f"{d}:{self.role_for_date(d)}")
        if bad:
            raise ValueError(f"Dates outside allowed roles for {purpose}: {bad[:8]}; allowed={sorted(allowed)}")

    def _date_in_any_role(self, value: str | date | datetime, roles: set[str]) -> bool:
        ranges = self._role_ranges()
        return any(ranges[role].contains(value) for role in roles if role in ranges)

    def _role_ranges(self) -> dict[str, DateRange]:
        return {
            "source_train": self.source_train,
            "source_fit": self.source_fit,
            "source_val": self.source_val,
            "source_test": self.source_test,
            "target_train": self.target_train,
            "target_adapt": self.target_adapt,
            "target_adaptation": self.target_adapt,
            "target_context": self.target_context,
            "target_support": self.target_context,
            "target_val": self.target_val,
            "target_eval": self.target_eval,
            "target_query": self.target_query,
        }

    @staticmethod
    def _expand_role_aliases(roles: Sequence[str]) -> set[str]:
        expanded = set(roles)
        if "source_train" in expanded:
            expanded.add("source_fit")
        if "source_fit" in expanded:
            expanded.add("source_train")
        if "target_context" in expanded or "target_support" in expanded or "target_adaptation" in expanded or "target_adapt" in expanded:
            expanded.add("target_train")
        if "target_train" in expanded:
            expanded.update({"target_context", "target_support", "target_adaptation", "target_adapt"})
        if "target_val" in expanded:
            expanded.add("source_val")
        if "target_eval" in expanded:
            expanded.add("target_query")
        if "target_query" in expanded:
            expanded.add("target_eval")
        # source_test shares the same date range as target_eval (2023-2025)
        if "source_test" in expanded:
            expanded.add("target_eval")
        if "target_eval" in expanded:
            expanded.add("source_test")
        return expanded
