"""Leakage guard utilities for HydroDA-OOD / HyperDA V4.4 zero/few-shot protocol."""
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
        roles = ["source_train", "source_fit", "source_val", "target_context"]
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
        """Validate labeled target support cycles for K-shot adaptation."""
        self.protocol.assert_dates_within(times, ["target_support"], "support_selection")

    def check_target_adaptation_scope(
        self,
        times: Iterable,
        *,
        purpose: str,
        labels_allowed: bool,
    ) -> None:
        """Validate target-domain adaptation/generalization data access.

        Main target adaptation may use only preregistered target_support cycles
        from 2015-2021 and must never touch target_val/target_eval for
        selection, early stopping, or gain calibration.
        """
        if purpose in {"early_stopping", "model_selection", "hyperparameter_selection", "checkpoint_selection"}:
            raise ValueError(
                f"{purpose} must not be driven by target labels in the main protocol. "
                "Use source_val_preregistered only."
            )
        self.protocol.assert_dates_within(times, ["target_support"], purpose)
        if not labels_allowed:
            self.check_label_access(times, purpose=purpose)

    def check_model_selection_scope(
        self,
        times: Iterable,
        *,
        purpose: str,
        allow_target_train: bool = False,
        model_selection_source: str = "source_val_preregistered",
    ) -> None:
        """Validate checkpoint/early-stopping/hyperparameter selection dates."""
        if model_selection_source != "source_val_preregistered":
            raise ValueError(
                "Main zero/few-shot protocol requires model_selection_source="
                f"'source_val_preregistered'; got {model_selection_source!r}. "
                "target_val is unused in the main protocol."
            )
        roles = ["source_val"]
        if allow_target_train:
            roles.append("target_support")
        self.protocol.assert_dates_within(times, roles, purpose)

    def check_target_side_selection_scope(self, times: Iterable, *, purpose: str) -> None:
        """Reject target-side validation/early-stopping/model selection in the main protocol."""
        self.protocol.assert_dates_within(times, ["target_val"], purpose)
        raise ValueError(
            f"{purpose} cannot use target_val in the main zero/few-shot protocol; "
            "target_val_usage=unused_in_main_protocol."
        )

    def check_target_residual_gain_calibration_scope(self, times: Iterable, *, purpose: str) -> None:
        """Reject target-label residual-gain calibration in the paper-facing protocol."""
        self.protocol.assert_dates_within(times, ["target_support", "target_context"], purpose)
        raise ValueError(
            f"target-label residual gain calibration is legacy/internal only for {purpose}; "
            "main protocol uses fixed preregistered source_val selection."
        )

    def check_query_evaluation_only(self, times: Iterable) -> None:
        self.protocol.assert_dates_within(times, ["target_eval"], "final_evaluation")

    def assert_method_table_allowed(self, method: str, table: str) -> None:
        from hydroda.baselines.registry import assert_allowed_for_table

        assert_allowed_for_table(method, table)
