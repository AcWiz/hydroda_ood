"""Leakage guard utilities for HydroDA-OOD / HyperDA V4 full-target-training protocol."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from hydroda.data.protocol import ProtocolConfig


FORBIDDEN_TARGET_EVAL_LABEL_PURPOSES = {
    "prompt_construction",
    "normalization",
    "support_selection",
    "target_train_selection",
    "target_adaptation",
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
        roles = ["source_train", "source_fit", "source_val", "target_train"]
        if allow_target_query_inputs:
            # Only for explicitly marked transductive input-only ablation, never main protocol.
            roles.append("target_eval")
        self.protocol.assert_dates_within(times, roles, "prompt_construction")

    def check_label_access(self, times: Iterable, *, purpose: str) -> None:
        if purpose in FORBIDDEN_TARGET_EVAL_LABEL_PURPOSES:
            self.protocol.assert_no_query_dates(times, purpose)

    def check_normalization_scope(self, times: Iterable, *, scope_name: str) -> None:
        if scope_name not in ("source_train_only", "source_fit_only"):
            raise ValueError(
                "HyperDA main protocol requires source_train_only or source_fit_only normalization. "
                f"Got scope={scope_name!r}."
            )
        allowed_roles = ["source_train", "source_fit"]
        self.protocol.assert_dates_within(times, allowed_roles, "normalization")

    def check_support_dates(self, times: Iterable) -> None:
        """Legacy few-shot support-date guard.

        2022 target support dates are now an alias of target_train dates and
        should only appear in explicitly marked few-shot ablations.
        """
        self.protocol.assert_dates_within(times, ["target_train"], "support_selection")

    def check_target_adaptation_scope(
        self,
        times: Iterable,
        *,
        purpose: str,
        labels_allowed: bool,
    ) -> None:
        """Validate target-domain adaptation/generalization data access.

        Main target adaptation may use target_train labels (the full 2015-2021
        historical target training period) but must never touch target_eval/query dates.
        """
        if purpose in {"early_stopping", "model_selection", "hyperparameter_selection", "checkpoint_selection"}:
            raise ValueError(
                f"{purpose} must not be driven by target_train labels in the main protocol. "
                "Use source_val or a pre-registered target-train internal validation rule."
            )
        self.protocol.assert_dates_within(times, ["target_train"], purpose)
        if not labels_allowed:
            self.check_label_access(times, purpose=purpose)

    def check_model_selection_scope(
        self,
        times: Iterable,
        *,
        purpose: str,
        allow_target_train: bool = False,
    ) -> None:
        """Validate checkpoint/early-stopping/hyperparameter selection dates."""
        roles = ["source_val"]
        if allow_target_train:
            roles.append("target_train")
        self.protocol.assert_dates_within(times, roles, purpose)

    def check_query_evaluation_only(self, times: Iterable) -> None:
        self.protocol.assert_dates_within(times, ["target_eval"], "final_evaluation")

    def assert_method_table_allowed(self, method: str, table: str) -> None:
        from hydroda.baselines.registry import assert_allowed_for_table

        assert_allowed_for_table(method, table)
