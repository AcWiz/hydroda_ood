"""Leakage guard utilities for HydroDA-OOD / HyperDA V4."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from hydroda.data.protocol import ProtocolConfig


FORBIDDEN_QUERY_LABEL_PURPOSES = {
    "prompt_construction",
    "normalization",
    "support_selection",
    "early_stopping",
    "model_selection",
    "threshold_calibration",
    "prompt_feature_tuning",
    "region_definition",
}


@dataclass
class LeakageGuard:
    protocol: ProtocolConfig

    def check_prompt_times(self, times: Iterable, *, allow_target_query_inputs: bool = False) -> None:
        roles = ["source_train", "source_fit", "source_val", "target_context"]
        if allow_target_query_inputs:
            # Only for explicitly marked transductive input-only ablation, never main V4.
            roles.append("target_query")
        self.protocol.assert_dates_within(times, roles, "prompt_construction")

    def check_label_access(self, times: Iterable, *, purpose: str) -> None:
        if purpose in FORBIDDEN_QUERY_LABEL_PURPOSES:
            self.protocol.assert_no_query_dates(times, purpose)

    def check_normalization_scope(self, times: Iterable, *, scope_name: str) -> None:
        if scope_name not in ("source_train_only", "source_fit_only"):
            raise ValueError(
                "HyperDA V4 main protocol requires source_train_only or source_fit_only normalization. "
                f"Got scope={scope_name!r}."
            )
        allowed_roles = ["source_train", "source_fit"]
        self.protocol.assert_dates_within(times, allowed_roles, "normalization")

    def check_support_dates(self, times: Iterable) -> None:
        self.protocol.assert_dates_within(times, ["target_context"], "support_selection")

    def check_query_evaluation_only(self, times: Iterable) -> None:
        self.protocol.assert_dates_within(times, ["target_query"], "final_evaluation")

    def assert_method_table_allowed(self, method: str, table: str) -> None:
        from hydroda.baselines.registry import assert_allowed_for_table

        assert_allowed_for_table(method, table)
