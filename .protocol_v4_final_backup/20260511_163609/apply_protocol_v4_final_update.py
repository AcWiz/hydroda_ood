#!/usr/bin/env python3
"""
Apply HydroDA-OOD / HyperDA Protocol V4-final update.

New frozen protocol:
  source_fit:     2015-01-01 .. 2020-12-31
  source_val:     2021-01-01 .. 2021-12-31   (source continents only)
  target_context: 2022-01-01 .. 2022-12-31   (K-cycle calibration only)
  target_query:   2023-01-01 .. 2025-12-31   (final evaluation only)

Run from the hydroda_ood repository root:
  python apply_protocol_v4_final_update.py

The script creates timestamped backups under .protocol_v4_final_backup/ before editing.
"""
from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path.cwd()
BACKUP_ROOT = ROOT / ".protocol_v4_final_backup" / datetime.now().strftime("%Y%m%d_%H%M%S")

TEXT_EXTS = {".md", ".yaml", ".yml", ".py", ".json", ".toml", ".txt"}
SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".protocol_v4_final_backup",
}
# Do not rewrite historical results/reports by default. They are records of old runs.
SKIP_TOP_LEVEL = {"artifacts", "wandb"}

@dataclass
class Change:
    path: str
    description: str

changes: list[Change] = []


def ensure_repo_root() -> None:
    required = ["CLAUDE.md", "完整研究计划方案.md", "context", "specs", "hydroda"]
    missing = [p for p in required if not (ROOT / p).exists()]
    if missing:
        raise SystemExit(
            "This script must be run from the hydroda_ood repository root. "
            f"Missing: {missing}"
        )


def should_process(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    if rel.parts and rel.parts[0] in SKIP_TOP_LEVEL:
        return False
    if any(part in SKIP_DIRS for part in rel.parts):
        return False
    return path.is_file() and path.suffix in TEXT_EXTS


def backup(path: Path) -> None:
    rel = path.relative_to(ROOT)
    dst = BACKUP_ROOT / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dst)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write(path: Path, text: str, description: str) -> None:
    old = read(path) if path.exists() else None
    if old == text:
        return
    if path.exists():
        backup(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    changes.append(Change(str(path.relative_to(ROOT)), description))


def update_file(path: str, transform, description: str) -> None:
    p = ROOT / path
    if not p.exists():
        return
    old = read(p)
    new = transform(old)
    if new != old:
        write(p, new, description)


def replace_many(text: str, replacements: list[tuple[str, str]]) -> str:
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def global_protocol_replacements() -> None:
    replacements = [
        ("Source train:               2015-01-01 to 2020-12-31\nTarget context/calibration:  2021-01-01 to 2021-12-31\nTarget query/evaluation:     2022-01-01 to 2025-12-31",
         "Source fit/train:           2015-01-01 to 2020-12-31\nSource validation:          2021-01-01 to 2021-12-31\nTarget context/calibration: 2022-01-01 to 2022-12-31\nTarget query/evaluation:    2023-01-01 to 2025-12-31"),
        ("Source train: 2015-2020\nTarget context / calibration: 2021\nTarget query / evaluation: 2022-2025",
         "Source fit/train: 2015-2020\nSource validation: 2021\nTarget context / calibration: 2022\nTarget query / evaluation: 2023-2025"),
        ("source_train:    2015-2020\n    target_context:  2021\n    target_query:    2022-2025",
         "source_fit:      2015-2020\n    source_val:      2021\n    target_context:  2022\n    target_query:    2023-2025"),
        ("source_train:   2015-01-01 到 2020-12-31\n    target_support: 2021-01-01 到 2021-12-31\n    target_query:   2022-01-01 到 2025-12-31",
         "source_fit:     2015-01-01 到 2020-12-31\n    source_val:     2021-01-01 到 2021-12-31（仅 source continents，用于 checkpoint / hyperparameter selection）\n    target_support: 2022-01-01 到 2022-12-31（target context / K-cycle calibration）\n    target_query:   2023-01-01 到 2025-12-31（final evaluation only）"),
        ("K = number of labeled target DA analysis cycles from 2021",
         "K = number of labeled target DA analysis cycles from target context year 2022"),
        ("one labeled DA cycle sampled from each season in 2021",
         "one labeled DA cycle sampled from each season in 2022"),
        ("one labeled DA cycle sampled from each month in 2021",
         "one labeled DA cycle sampled from each month in 2022"),
        ("HyperDA-Zero：K=0，只用 2021 input-side spatial/temporal prompt。",
         "HyperDA-Zero：K=0，不使用 target analysis labels；仅允许使用 target context year 2022 的 input-side spatial/temporal prompt。"),
        ("target_context:  2021", "target_context:  2022"),
        ("target_query:    2022-2025", "target_query:    2023-2025"),
        ("target_context/calibration 和 target query", "target_context/calibration 和 target query"),
        ("support_year: 2021", "support_year: 2022"),
        ("query_years: [2022, 2023, 2024, 2025]", "query_years: [2023, 2024, 2025]"),
        ("one_valid_cycle_per_season_in_2021", "one_valid_cycle_per_season_in_2022"),
        ("one_valid_cycle_per_month_in_2021", "one_valid_cycle_per_month_in_2022"),
        ("Time availability in 2021", "Time availability in 2022"),
        ("target 2021 input-only stream", "target 2022 input-only stream"),
        ("Get available support dates in 2021", "Get available support dates in 2022"),
        ("Available support dates in 2021", "Available support dates in 2022"),
        ("Support (2021)", "Target context/support (2022)"),
        ("Query (2022-2025)", "Target query (2023-2025)"),
        ("support_mask = years == 2021", "support_mask = years == 2022"),
        ("query_mask = (years >= 2022) & (years <= 2025)", "query_mask = (years >= 2023) & (years <= 2025)"),
        ("--k-values", "--k-values"),
        ("default=[0, 4, 12, 24]", "default=[0, 4, 12]"),
        ("K values (0, 4, 12, 24)", "K values (0, 4, 12); K=24 is optional internal ablation only"),
        ("Must be one of {0, 4, 12, 24}", "Must be one of {0, 4, 12}; K=24 is optional internal ablation only"),
        ("K: Number of support dates (0, 4, 12, or 24)", "K: Number of support dates for main experiments (0, 4, or 12)"),
        ("test_support_dates_in_2021_only", "test_support_dates_in_2022_only"),
        ("test_query_dates_after_2022_only", "test_query_dates_after_2023_only"),
        ("fall within 2021", "fall within 2022"),
        ("test_all_support_dates_in_2021", "test_all_support_dates_in_2022"),
        ("assert year == 2021", "assert year == 2022"),
        ("not in 2021", "not in 2022"),
        ("fall within 2022-2025", "fall within 2023-2025"),
        ("test_all_query_dates_in_2022_to_2025", "test_all_query_dates_in_2023_to_2025"),
        ("valid_years = {2022, 2023, 2024, 2025}", "valid_years = {2023, 2024, 2025}"),
        ("not in 2022-2025", "not in 2023-2025"),
        ("test_target_query_period_is_2022_onwards", "test_target_query_period_is_2023_onwards"),
        ("assert year >= 2022", "assert year >= 2023"),
        ("year {year} < 2022", "year {year} < 2023"),
        ("guard.check_normalization_scope([\"2022-01-01\"]", "guard.check_normalization_scope([\"2023-01-01\"]"),
        ("guard.check_support_dates([\"2021-03-01\", \"2021-09-01\"])", "guard.check_support_dates([\"2022-03-01\", \"2022-09-01\"])")
    ]

    for path in ROOT.rglob("*"):
        if not should_process(path):
            continue
        old = read(path)
        new = replace_many(old, replacements)
        if new != old:
            write(path, new, "global Protocol V4-final year/split replacement")


def update_claude_md() -> None:
    def transform(text: str) -> str:
        new_section = """## 5. 时间协议与 K-cycle calibration

冻结时间协议（Protocol V4-final）：

```text
Source fit/train:           2015-01-01 to 2020-12-31
Source validation:          2021-01-01 to 2021-12-31
Target context/calibration: 2022-01-01 to 2022-12-31
Target query/evaluation:    2023-01-01 to 2025-12-31
```

用途冻结：

```text
source_fit:     训练 shared backbone、source-only model、source operator episode bank
source_val:     只用于 checkpoint selection、early stopping、hyperparameter / architecture selection；必须来自 source continents
target_context: 只用于 K-cycle target calibration / adaptation / calibration prompt summary；不能用于 checkpoint 或超参数选择
target_query:   只用于最终 offline evaluation；任何 target query labels 不得进入训练、校准、normalization、prompt construction 或 model selection
```

K 正式定义为：

```text
K = number of labeled target DA analysis cycles from target context year 2022
```

主实验 K 取值：

```text
K ∈ {0, 4, 12}
```

支持集采样：

```text
K=0:  no target analysis labels; input-side prompt only
K=4:  one labeled DA cycle sampled from each season in 2022
K=12: one labeled DA cycle sampled from each month in 2022
```

K 不是 patches、pixels 或 mini-batches 数。同一天切出多少 spatial patches，都只算一个 DA calibration cycle。

---"""
        pattern = r"## 5\. 时间协议与 K-cycle calibration\n.*?---\n## 6\. Leakage Guard"
        replaced = re.sub(pattern, new_section + "\n## 6. Leakage Guard", text, flags=re.S)
        return replaced
    update_file("CLAUDE.md", transform, "rewrite frozen time protocol section")


def update_research_plan() -> None:
    def transform(text: str) -> str:
        new_section = """## 5. 主实验协议

Leave-one-continent-out：

```text
US + CN -> AU
US + AU -> CN
CN + AU -> US
```

Protocol V4-final 时间协议：

```text
Source fit/train:           2015-2020
Source validation:          2021
Target context / calibration: 2022
Target query / evaluation:    2023-2025
```

`source_val=2021` 只能来自 source continents，用于 checkpoint selection、early stopping 和 hyperparameter / architecture selection。`target_context=2022` 只能用于 K-cycle target calibration / adaptation / calibration prompt summary，不能用于模型选择或调参。`target_query=2023-2025` 只用于最终 evaluation。

K-cycle calibration：

```text
K ∈ {0, 4, 12}
```

K 是 target context year 2022 中的 DA cycles，不是 patches。"""
        pattern = r"## 5\. 主实验协议\n.*?## 6\. Leakage Guard"
        replaced = re.sub(pattern, new_section + "\n## 6. Leakage Guard", text, flags=re.S)
        return replaced
    update_file("完整研究计划方案.md", transform, "rewrite main experimental protocol section")


def write_protocol_yaml() -> None:
    text = """protocol_name: HydroDA-OOD-HyperDA-V4
protocol_version: v4.1-final
protocol_freeze_id: hyperda_v4_final_2015_2025_context2022_query2023_2025_k0_4_12
objective: neural_land_DA_increment_emulation
not_objective:
  - generic_soil_moisture_prediction
  - true_soil_moisture_estimation

periods:
  # Backward-compatible alias used by some existing scripts.
  # Paper-facing text should call this source_fit/train.
  source_train:
    start: '2015-01-01'
    end: '2020-12-31'
    alias_of: source_fit
    allowed_usage:
      - training_shared_backbone
      - training_source_only_backbone
      - training_source_operator_episode_bank
  source_fit:
    start: '2015-01-01'
    end: '2020-12-31'
    allowed_usage:
      - training_shared_backbone
      - training_source_only_backbone
      - training_source_operator_episode_bank
  source_val:
    start: '2021-01-01'
    end: '2021-12-31'
    domain_constraint: source_continents_only
    allowed_usage:
      - checkpoint_selection
      - early_stopping
      - hyperparameter_selection
      - architecture_selection
    forbidden_usage:
      - target_calibration
      - final_evaluation
  target_context:
    start: '2022-01-01'
    end: '2022-12-31'
    domain_constraint: held_out_target_continent_only
    allowed_usage:
      - K_cycle_calibration
      - target_adaptation
      - calibration_prompt_summary
    forbidden_usage:
      - checkpoint_selection
      - early_stopping
      - hyperparameter_selection
      - architecture_selection
      - model_selection
  target_query:
    start: '2023-01-01'
    end: '2025-12-31'
    domain_constraint: held_out_target_continent_only
    allowed_usage:
      - final_offline_evaluation_only
    forbidden_usage:
      - training
      - target_calibration
      - prompt_construction
      - normalization
      - support_selection
      - early_stopping
      - model_selection
      - threshold_calibration
      - prompt_feature_tuning
      - region_definition

K_values_main: [0, 4, 12]
K_definition: number_of_labeled_target_DA_analysis_cycles_from_target_context_2022
support_sampling:
  support_year: 2022
  K0: no_target_analysis_labels_input_side_prompt_only
  K4: one_valid_cycle_per_season_in_2022
  K12: one_valid_cycle_per_month_in_2022
  seeds_main: 5
  seeds_preferred_final: 10

leave_one_continent_out:
  - source: [US, CN]
    target: AU
  - source: [US, AU]
    target: CN
  - source: [CN, AU]
    target: US

leakage_forbidden:
  - target_query_labels_for_prompt_construction
  - target_query_labels_for_normalization
  - target_query_labels_for_support_selection
  - target_query_labels_for_early_stopping
  - target_query_labels_for_model_selection
  - target_query_labels_for_threshold_calibration
  - target_query_labels_for_prompt_feature_tuning
  - target_context_labels_for_checkpoint_or_hyperparameter_selection
  - analysis_increments_for_region_definition
  - model_errors_for_region_definition

paper_protocol_note: >-
  For each held-out target continent, train on source-domain years 2015-2020,
  select checkpoints and hyperparameters only on source-domain year 2021,
  optionally calibrate using K labeled target DA cycles from target year 2022,
  and report final metrics exclusively on target years 2023-2025.
"""
    write(ROOT / "specs/protocol_v4.yaml", text, "overwrite Protocol V4-final YAML spec")


def write_kdate_yaml() -> None:
    text = """version: K-date Few-Cycle Calibration V4-final
protocol_freeze_id: hyperda_v4_final_2015_2025_context2022_query2023_2025_k0_4_12

K_values_main: [0, 4, 12]
K_values_supported_internal: [0, 4, 12, 24]
K24_status: optional_internal_ablation_not_main_table

source_fit_years: [2015, 2016, 2017, 2018, 2019, 2020]
source_val_year: 2021
support_year: 2022
query_years: [2023, 2024, 2025]

definitions_cn:
  K: 目标区域 target context year 2022 可用的 labeled DA analysis dates/cycles 数量
  not_K: 不是 patch 数，也不是 pixel 数，也不是 mini-batch 数
  source_val: 只能来自 source continents，用于 checkpoint / hyperparameter / architecture selection
  target_context: 只能来自 held-out target continent 的 2022 年，用于 K-cycle calibration / adaptation
  target_query: 只能来自 held-out target continent 的 2023-2025 年，用于最终评估

selection:
  K0:
    target_analysis_labels_allowed: false
    target_input_only_allowed: true
  K4:
    northern_hemisphere_bins:
      - [1, 2, 3]
      - [4, 5, 6]
      - [7, 8, 9]
      - [10, 11, 12]
    southern_hemisphere_bins:
      - [7, 8, 9]
      - [10, 11, 12]
      - [1, 2, 3]
      - [4, 5, 6]
    dates_per_bin: 1
  K12:
    dates_per_month: 1
  K24_optional_internal:
    dates_per_month_half:
      first_half_days: [1, 15]
      second_half_days: [16, 31]
    dates_per_half_month: 1

seeds:
  minimum: 5
  preferred: 10

rules_cn:
  same_support_dates_across_methods: 同一 continent/region/K/seed 下所有方法必须使用相同 support dates
  no_query_labels_for_support_selection: 不允许用 target query labels 选择 support
  no_target_context_for_model_selection: target context labels 不能用于 checkpoint、超参数或模型选择
  fail_if_not_enough_valid_dates: support dates 不足时必须报错或显式记录 degraded split
  log_effective_support_budget: 必须记录 K dates × patches/date × valid pixels
"""
    write(ROOT / "specs/kdate_protocol.yaml", text, "overwrite K-date Protocol V4-final YAML spec")


def update_protocol_py() -> None:
    def transform(text: str) -> str:
        text = text.replace(
            'protocol_freeze_id: str = "hyperda_v4_2015_2025_k0_4_12"',
            'protocol_freeze_id: str = "hyperda_v4_final_2015_2025_context2022_query2023_2025_k0_4_12"',
        )
        text = text.replace(
            'DateRange.from_strings("source_fit", "2015-01-01", "2019-12-31")',
            'DateRange.from_strings("source_fit", "2015-01-01", "2020-12-31")',
        )
        text = text.replace(
            'DateRange.from_strings("source_val", "2020-01-01", "2020-12-31")',
            'DateRange.from_strings("source_val", "2021-01-01", "2021-12-31")',
        )
        text = text.replace(
            'DateRange.from_strings("target_context", "2021-01-01", "2021-12-31")',
            'DateRange.from_strings("target_context", "2022-01-01", "2022-12-31")',
        )
        text = text.replace(
            'DateRange.from_strings("target_query", "2022-01-01", "2025-12-31")',
            'DateRange.from_strings("target_query", "2023-01-01", "2025-12-31")',
        )
        return text
    update_file("hydroda/data/protocol.py", transform, "update ProtocolConfig date ranges and freeze id")


def update_protocol_guard_tests() -> None:
    def transform(text: str) -> str:
        text = text.replace('assert p.role_for_date("2020-06-01") == "source_val"',
                            'assert p.role_for_date("2020-06-01") == "source_fit"')
        if 'assert p.role_for_date("2021-06-01") == "source_val"' not in text:
            text = text.replace('assert p.role_for_date("2021-06-01") == "target_context"',
                                'assert p.role_for_date("2021-06-01") == "source_val"')
            text = text.replace('assert p.role_for_date("2023-06-01") == "target_query"',
                                'assert p.role_for_date("2022-06-01") == "target_context"\n    assert p.role_for_date("2023-06-01") == "target_query"')
        text = text.replace('def test_guard_accepts_support_dates_in_2021():',
                            'def test_guard_accepts_support_dates_in_2022():')
        text = text.replace('guard.check_support_dates(["2021-03-01", "2021-09-01"])',
                            'guard.check_support_dates(["2022-03-01", "2022-09-01"])')
        text = text.replace('guard.check_normalization_scope(["2022-01-01"]',
                            'guard.check_normalization_scope(["2023-01-01"]')
        return text
    update_file("tests/test_protocol_leakage_guard.py", transform, "update ProtocolConfig tests for V4-final")


def update_context_map_and_add_note() -> None:
    note = """# 12_PROTOCOL_V4_FINAL_UPDATE.md — Protocol V4-final 冻结说明

本文件记录 2026-05-11 冻结的新时间协议。若旧文档、旧任务、旧 artifacts 中仍出现 2021 target context 或 2022-2025 target query，以本文件、`CLAUDE.md`、`完整研究计划方案.md`、`specs/protocol_v4.yaml` 为准。

## 冻结协议

```text
source_fit:     2015-01-01 to 2020-12-31
source_val:     2021-01-01 to 2021-12-31  source continents only
target_context: 2022-01-01 to 2022-12-31  held-out target continent; K-cycle calibration only
target_query:   2023-01-01 to 2025-12-31  final offline evaluation only
```

## 关键纪律

`source_val=2021` 只能来自 source continents，用于 checkpoint selection、early stopping、hyperparameter selection 和 architecture selection。`target_context=2022` 只能来自 held-out target continent，用于 K-cycle calibration、adapter/LoRA/HyperDA-Calib/HyperDA-Refine 的 target adaptation 或 calibration prompt summary。`target_context` 不能用于 checkpoint 或超参数选择。`target_query=2023-2025` 只能用于最终评估，不能用于 prompt construction、normalization、support-date selection、training、early stopping、model selection、threshold calibration 或 region definition。

## K 定义

K 是 target context year 2022 中的 labeled DA analysis cycles 数量，不是 patch 数、pixel 数或 mini-batch 数。主实验只使用 `K ∈ {0, 4, 12}`；K=24 若保留，只能作为 optional internal ablation，不能进入主表。
"""
    write(ROOT / "context/12_PROTOCOL_V4_FINAL_UPDATE.md", note, "add Protocol V4-final context note")

    cmap = ROOT / "context/00_EXECUTABLE_CONTEXT_MAP.md"
    if cmap.exists():
        text = read(cmap)
        if "12_PROTOCOL_V4_FINAL_UPDATE.md" not in text:
            addition = """

## Protocol V4-final update

- `context/12_PROTOCOL_V4_FINAL_UPDATE.md`：冻结新时间协议 `source_fit=2015-2020`、`source_val=2021`、`target_context=2022`、`target_query=2023-2025`。
- `specs/protocol_v4.yaml`：机器可读的 Protocol V4-final single source of truth。
- `specs/kdate_protocol.yaml`：K-cycle support-year 与 query-year 的机器可读约束。
"""
            write(cmap, text.rstrip() + addition + "\n", "register Protocol V4-final context file")


def add_task_doc() -> None:
    text = """# phase2_protocol_v4_final_update.md — 更新 Protocol V4-final

## 目标

将项目从旧时间协议：

```text
source_train: 2015-2020
target_context: 2021
target_query: 2022-2025
```

更新为新冻结协议：

```text
source_fit:     2015-2020
source_val:     2021
target_context: 2022
target_query:   2023-2025
```

## 必改文件

- `CLAUDE.md`
- `完整研究计划方案.md`
- `context/01_RESEARCH_CONTRACT.md`
- `context/04_KDATE_SPLIT_PROTOCOL.md`
- `context/00_EXECUTABLE_CONTEXT_MAP.md`
- `specs/protocol_v4.yaml`
- `specs/kdate_protocol.yaml`
- `hydroda/data/protocol.py`
- `scripts/build_kdate_splits.py`
- `hydroda/splits/kdate.py`
- 相关 tests：`test_protocol_leakage_guard.py`、`test_kdate_splits_no_leakage.py`、`test_target_query_evaluation_only.py`

## 验收标准

1. `ProtocolConfig().role_for_date("2020-06-01") == "source_fit"`。
2. `ProtocolConfig().role_for_date("2021-06-01") == "source_val"`。
3. `ProtocolConfig().role_for_date("2022-06-01") == "target_context"`。
4. `ProtocolConfig().role_for_date("2023-06-01") == "target_query"`。
5. split builder 使用 2022 年作为 support/context year，2023-2025 作为 query years。
6. target context labels 不能用于 checkpoint、early stopping、hyperparameter 或 model selection。
7. target query labels 仍然只能用于最终 offline evaluation。

## 建议测试

```bash
pytest tests/test_protocol_leakage_guard.py \
       tests/test_kdate_splits_no_leakage.py \
       tests/test_target_query_evaluation_only.py -q
```

如果没有真实数据或 split artifacts，数据依赖型测试可以 skip；`test_protocol_leakage_guard.py` 必须通过。
"""
    write(ROOT / "tasks/phase2_protocol_v4_final_update.md", text, "add task doc for Protocol V4-final migration")


def audit_leftovers() -> list[str]:
    patterns = [
        "target_context:  2021",
        "Target context/calibration:  2021",
        "Target context / calibration: 2021",
        "support_year: 2021",
        "support_mask = years == 2021",
        "target_query:    2022-2025",
        "Query (2022-2025)",
        "test_all_support_dates_in_2021",
        "test_target_query_period_is_2022_onwards",
        "guard.check_support_dates([\"2021",
    ]
    leftovers: list[str] = []
    for path in ROOT.rglob("*"):
        if not should_process(path):
            continue
        text = read(path)
        for pat in patterns:
            if pat in text:
                leftovers.append(f"{path.relative_to(ROOT)}: contains {pat!r}")
    return leftovers


def main() -> None:
    ensure_repo_root()
    BACKUP_ROOT.mkdir(parents=True, exist_ok=True)

    # First broad replacements, then precise rewrites for authoritative files.
    global_protocol_replacements()
    update_claude_md()
    update_research_plan()
    write_protocol_yaml()
    write_kdate_yaml()
    update_protocol_py()
    update_protocol_guard_tests()
    update_context_map_and_add_note()
    add_task_doc()

    print("\nProtocol V4-final update complete.")
    print(f"Backups: {BACKUP_ROOT.relative_to(ROOT)}")
    print("\nModified / added files:")
    for c in changes:
        print(f"- {c.path}: {c.description}")

    leftovers = audit_leftovers()
    if leftovers:
        print("\nPotential leftover old-protocol strings to review manually:")
        for item in leftovers[:80]:
            print(f"- {item}")
        if len(leftovers) > 80:
            print(f"... {len(leftovers) - 80} more")
        print("\nNote: historical artifacts/reports under artifacts/ and wandb/ were intentionally skipped.")
    else:
        print("\nNo critical old-protocol strings found outside skipped historical directories.")

    print("\nSuggested checks:")
    print("  python -m pytest tests/test_protocol_leakage_guard.py -q")
    print("  python -m pytest tests/test_kdate_splits_no_leakage.py tests/test_target_query_evaluation_only.py -q")
    print("  git diff -- CLAUDE.md 完整研究计划方案.md context specs hydroda scripts tests tasks")


if __name__ == "__main__":
    main()
