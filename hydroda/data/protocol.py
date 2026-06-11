"""Protocol objects for HydroDA-OOD / HyperDA V4.4 zero/few-shot generalization."""
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

    The paper-facing protocol is zero/few-shot target generalization. Source
    training fits on 2015-2021 and source validation in 2022 is the only main
    checkpoint / hyperparameter selection source. Target-domain 2015-2021
    input-side context may build a prompt. For K in {0, 4, 12}, only K labeled
    target support cycles may update target-specific lightweight variables.
    Target validation is unused in the main protocol, and target evaluation
    labels are final-offline-evaluation-only for 2023-2025.
    """

    protocol_name: str = "HydroDA-OOD-HyperDA-V4"
    protocol_freeze_id: str = "hyperda_v4_4_zero_few_shot_generalization_2015_2025_context2015_2021_sourceval2022_eval2023_2025"
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
    target_context: DateRange = field(
        default_factory=lambda: DateRange.from_strings("target_context", "2015-01-01", "2021-12-31")
    )
    target_support: DateRange = field(
        default_factory=lambda: DateRange.from_strings("target_support", "2015-01-01", "2021-12-31")
    )
    # Backward-compatible legacy full-target aliases. They share dates with
    # target_context/support but are not paper-facing main settings.
    target_train: DateRange = field(
        default_factory=lambda: DateRange.from_strings("target_train", "2015-01-01", "2021-12-31")
    )
    target_adapt: DateRange = field(
        default_factory=lambda: DateRange.from_strings("target_adapt", "2015-01-01", "2021-12-31")
    )
    target_val: DateRange = field(
        default_factory=lambda: DateRange.from_strings("target_val", "2022-01-01", "2022-12-31")
    )
    target_eval: DateRange = field(
        default_factory=lambda: DateRange.from_strings("target_eval", "2023-01-01", "2025-12-31")
    )
    # Backward-compatible date-range alias for older V4 docs/tests. The role
    # returned by role_for_date is target_eval.
    target_query: DateRange = field(
        default_factory=lambda: DateRange.from_strings("target_query", "2023-01-01", "2025-12-31")
    )
    main_adaptation_settings: Sequence[str] = ("zero_shot_context", "few_shot_k4", "few_shot_k12")
    main_K_values: Sequence[int] = (0, 4, 12)
    legacy_full_target_train_settings: Sequence[str] = ("target_full_train",)
    # Backward-compatible alias for older callers; these values are now the
    # main K-axis rather than legacy ablations.
    legacy_few_shot_K_values: Sequence[int] = (0, 4, 12)

    def role_for_date(self, value: str | date | datetime) -> str:
        if self.source_fit.contains(value):
            return "source_fit"
        if self.source_val.contains(value):
            return "source_val"
        if self.target_eval.contains(value):
            return "target_eval"
        return "outside_protocol"

    def assert_supported_adaptation_setting(
        self,
        setting: str,
        *,
        allow_legacy_full_target_train: bool = False,
    ) -> None:
        if setting in set(self.main_adaptation_settings):
            return
        if setting in set(self.legacy_full_target_train_settings):
            if allow_legacy_full_target_train:
                return
            raise ValueError(
                f"adaptation_setting={setting!r} is legacy/internal. "
                "Pass allow_legacy_full_target_train=True only for explicit historical reproduction."
            )
        raise ValueError(
            f"Unsupported main adaptation_setting={setting!r}. "
            f"Main protocol uses {list(self.main_adaptation_settings)}; "
            "target_full_train is legacy/internal only."
        )

    def assert_legacy_few_shot_K(self, K: int) -> None:
        self.assert_supported_K(K)

    def assert_supported_K(self, K: int, *, allow_legacy: bool = False) -> None:
        """Validate the paper-facing K-axis.

        ``allow_legacy`` is accepted for older call sites and has no effect for
        K in the current main axis.
        """
        if int(K) not in set(self.main_K_values):
            raise ValueError(
                f"Unsupported main K={K}. Main zero/few-shot protocol uses {list(self.main_K_values)}."
            )

    def adaptation_setting_for_K(self, K: int) -> str:
        self.assert_supported_K(K)
        return "zero_shot_context" if int(K) == 0 else f"few_shot_k{int(K)}"

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
            "target_support": self.target_support,
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
