# Phase 3A — Forecast-Only Baseline and Metrics Harness

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement forecast-only baseline (`pred_increment = 0`) and reusable metrics/evaluation harness for Phase 3–5.

**Architecture:** Pure numpy metric functions (no torch, no model) applied via a thin evaluation harness. `ForecastBaseline` is a stateless predictor. All masking uses `metric_mask > 0.5` filter. Output is long-format CSV.

**Tech Stack:** numpy, xarray (for NetCDF loading already in dataset), pandas (for CSV output), pytest

---

## Task 1: `hydroda/baselines/forecast.py` + `__init__.py`

**Files:**
- Create: `hydroda/baselines/__init__.py`
- Create: `hydroda/baselines/forecast.py`
- Test: `tests/test_forecast_baseline.py` (first 3 sanity tests)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_forecast_baseline.py
import numpy as np
import pytest
from hydroda.baselines.forecast import ForecastBaseline

def test_forecast_baseline_pred_increment_zero():
    """pred_increment_surface is all zeros."""
    baseline = ForecastBaseline()
    sample = {
        "forecast_surface": np.random.rand(256, 640).astype(np.float32),
        "forecast_rootzone": np.random.rand(256, 640).astype(np.float32),
    }
    result = baseline.predict(sample)
    assert result["pred_increment_surface"].shape == sample["forecast_surface"].shape
    assert np.all(result["pred_increment_surface"] == 0.0)
    assert np.all(result["pred_increment_rootzone"] == 0.0)

def test_pred_analysis_equals_forecast_for_forecast_baseline():
    """pred_analysis == forecast exactly for forecast-only baseline."""
    baseline = ForecastBaseline()
    forecast_s = np.random.rand(256, 640).astype(np.float32)
    forecast_rz = np.random.rand(256, 640).astype(np.float32)
    sample = {"forecast_surface": forecast_s, "forecast_rootzone": forecast_rz}
    result = baseline.predict(sample)
    np.testing.assert_array_equal(result["pred_analysis_surface"], forecast_s)
    np.testing.assert_array_equal(result["pred_analysis_rootzone"], forecast_rz)

def test_forecast_skill_is_zero():
    """analysis_skill_vs_forecast ≈ 0 for forecast-only (rtol=0.1, atol=0.05)."""
    from hydroda.metrics.skill import analysis_skill_vs_forecast
    baseline = ForecastBaseline()
    forecast_s = np.random.rand(256, 640).astype(np.float32)
    analysis_s = forecast_s + np.random.randn(256, 640).astype(np.float32) * 0.01
    sample = {"forecast_surface": forecast_s, "forecast_rootzone": forecast_rz}
    pred = baseline.predict(sample)
    skill = analysis_skill_vs_forecast(
        pred["pred_analysis_surface"], analysis_s, forecast_s,
        np.ones_like(forecast_s)
    )
    assert np.abs(skill) < 0.05, f"skill={skill} should be ≈ 0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_forecast_baseline.py::test_forecast_baseline_pred_increment_zero -v`
Expected: FAIL — `hydroda.baselines` has no `forecast` module

- [ ] **Step 3: Write minimal implementation**

```python
# hydroda/baselines/__init__.py
from .forecast import ForecastBaseline

__all__ = ["ForecastBaseline"]
```

```python
# hydroda/baselines/forecast.py
"""Forecast-only baseline: pred_increment = 0, pred_analysis = forecast."""
from __future__ import annotations
import numpy as np
from typing import Dict


class ForecastBaseline:
    """Forecast-only baseline: pred_increment = 0, pred_analysis = forecast.

    This is the sanity baseline — no fitting, no target data used.
    """

    def predict(self, sample: Dict) -> Dict:
        return {
            "pred_increment_surface": np.zeros_like(sample["forecast_surface"]),
            "pred_increment_rootzone": np.zeros_like(sample["forecast_rootzone"]),
            "pred_analysis_surface": sample["forecast_surface"].copy(),
            "pred_analysis_rootzone": sample["forecast_rootzone"].copy(),
        }

    @property
    def method_id(self) -> str:
        return "forecast_only"

    @property
    def method_name(self) -> str:
        return "Forecast-Only (Zero Increment)"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_forecast_baseline.py::test_forecast_baseline_pred_increment_zero tests/test_forecast_baseline.py::test_pred_analysis_equals_forecast_for_forecast_baseline tests/test_forecast_baseline.py::test_forecast_skill_is_zero -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hydroda/baselines/__init__.py hydroda/baselines/forecast.py tests/test_forecast_baseline.py
git commit -m "feat(phase3A): add ForecastBaseline predictor"
```

---

## Task 2: `hydroda/metrics/skill.py` + `__init__.py`

**Files:**
- Create: `hydroda/metrics/__init__.py`
- Create: `hydroda/metrics/skill.py`
- Test: add to `tests/test_forecast_baseline.py`

- [ ] **Step 1: Write failing test for mask compliance**

```python
# append to tests/test_forecast_baseline.py
def test_metrics_respect_metric_mask():
    """Metric values are zero/NaN outside metric_mask region."""
    from hydroda.metrics.skill import analysis_rmse
    mask = np.zeros((10, 10), dtype=np.float32)
    mask[3:7, 3:7] = 1.0
    pred = np.ones((10, 10), dtype=np.float32)
    true = np.zeros((10, 10), dtype=np.float32)
    rmse = analysis_rmse(pred, true, mask)
    # Inside mask: pred=1, true=0 → rmse = 1.0
    # Outside mask: should be ignored → valid pixels = 16
    assert np.isfinite(rmse) and rmse == 1.0
```

- [ ] **Step 2: Run test — verify it fails**

Run: `python -m pytest tests/test_forecast_baseline.py::test_metrics_respect_metric_mask -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write minimal implementation**

```python
# hydroda/metrics/__init__.py
from .skill import (
    analysis_rmse,
    analysis_mae,
    analysis_skill_vs_forecast,
    increment_rmse,
    increment_mae,
    increment_bias,
    increment_corr,
    sign_accuracy_deadzone,
    valid_pixel_count,
    effective_mask_fraction,
)

__all__ = [
    "analysis_rmse",
    "analysis_mae",
    "analysis_skill_vs_forecast",
    "increment_rmse",
    "increment_mae",
    "increment_bias",
    "increment_corr",
    "sign_accuracy_deadzone",
    "valid_pixel_count",
    "effective_mask_fraction",
]
```

```python
# hydroda/metrics/skill.py
"""Metric functions using numpy — no torch, no model dependencies."""
from __future__ import annotations
import numpy as np


def _apply_mask(arr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return 1D valid array and mask flattened for masked computation."""
    m = mask > 0.5
    return arr[m], m[m]


def valid_pixel_count(mask: np.ndarray) -> int:
    return int(np.sum(mask > 0.5))


def effective_mask_fraction(mask: np.ndarray) -> float:
    total = mask.size
    if total == 0:
        return np.nan
    return float(np.sum(mask > 0.5)) / total


def analysis_rmse(pred: np.ndarray, true: np.ndarray, mask: np.ndarray) -> float:
    p, m = _apply_mask(pred, mask)
    t, _ = _apply_mask(true, mask)
    if len(p) == 0:
        return np.nan
    return float(np.sqrt(np.mean((p - t) ** 2)))


def analysis_mae(pred: np.ndarray, true: np.ndarray, mask: np.ndarray) -> float:
    p, m = _apply_mask(pred, mask)
    t, _ = _apply_mask(true, mask)
    if len(p) == 0:
        return np.nan
    return float(np.mean(np.abs(p - t)))


def analysis_skill_vs_forecast(
    pred: np.ndarray, true: np.ndarray, forecast: np.ndarray, mask: np.ndarray
) -> float:
    """1 - rmse(pred, true) / rmse(forecast, true). Returns ~0 for forecast-only."""
    p, m = _apply_mask(pred, mask)
    t, _ = _apply_mask(true, mask)
    f, _ = _apply_mask(forecast, mask)
    if len(p) == 0:
        return np.nan
    rmse_pred = np.sqrt(np.mean((p - t) ** 2))
    rmse_fcst = np.sqrt(np.mean((f - t) ** 2))
    if rmse_fcst == 0:
        return np.nan
    return float(1 - rmse_pred / rmse_fcst)


def increment_rmse(pred_inc: np.ndarray, true_inc: np.ndarray, mask: np.ndarray) -> float:
    p, m = _apply_mask(pred_inc, mask)
    t, _ = _apply_mask(true_inc, mask)
    if len(p) == 0:
        return np.nan
    return float(np.sqrt(np.mean((p - t) ** 2)))


def increment_mae(pred_inc: np.ndarray, true_inc: np.ndarray, mask: np.ndarray) -> float:
    p, m = _apply_mask(pred_inc, mask)
    t, _ = _apply_mask(true_inc, mask)
    if len(p) == 0:
        return np.nan
    return float(np.mean(np.abs(p - t)))


def increment_bias(pred_inc: np.ndarray, true_inc: np.ndarray, mask: np.ndarray) -> float:
    p, m = _apply_mask(pred_inc, mask)
    t, _ = _apply_mask(true_inc, mask)
    if len(p) == 0:
        return np.nan
    return float(np.mean(p - t))


def increment_corr(pred_inc: np.ndarray, true_inc: np.ndarray, mask: np.ndarray) -> float:
    p, m = _apply_mask(pred_inc, mask)
    t, _ = _apply_mask(true_inc, mask)
    if len(p) < 2:
        return np.nan
    return float(np.corrcoef(p, t)[0, 1])


def sign_accuracy_deadzone(
    pred_inc: np.ndarray, true_inc: np.ndarray, mask: np.ndarray, epsilon: float = 0.005
) -> float:
    """Fraction where sign(pred_inc) == sign(true_inc) outside deadzone."""
    p, m = _apply_mask(pred_inc, mask)
    t, _ = _apply_mask(true_inc, mask)
    if len(p) == 0:
        return np.nan
    deadzone = np.abs(t) < epsilon
    outside = ~deadzone
    if np.sum(outside) == 0:
        return np.nan
    signs_match = np.sign(p[outside]) == np.sign(t[outside])
    return float(np.mean(signs_match))
```

- [ ] **Step 4: Run tests — verify they pass**

Run: `python -m pytest tests/test_forecast_baseline.py::test_metrics_respect_metric_mask -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hydroda/metrics/__init__.py hydroda/metrics/skill.py
git commit -m "feat(phase3A): add numpy metric functions"
```

---

## Task 3: `hydroda/evaluation/harness.py` + `__init__.py`

**Files:**
- Create: `hydroda/evaluation/__init__.py`
- Create: `hydroda/evaluation/harness.py`

- [ ] **Step 1: Write harness implementation**

```python
# hydroda/evaluation/__init__.py
from .harness import evaluate_split

__all__ = ["evaluate_split"]
```

```python
# hydroda/evaluation/harness.py
"""Evaluation harness — produces long-format per-sample metrics."""
from __future__ import annotations
from typing import Dict, List

import numpy as np

from hydroda.metrics.skill import (
    analysis_rmse,
    analysis_mae,
    analysis_skill_vs_forecast,
    increment_rmse,
    increment_mae,
    increment_bias,
    increment_corr,
    sign_accuracy_deadzone,
    valid_pixel_count,
    effective_mask_fraction,
)


def evaluate_split(
    dataset,
    predictor,
    split_role: str,
    experiment_id: str,
    protocol_freeze_id: str,
) -> List[Dict]:
    """Evaluate predictor on all samples in dataset, return per-sample metrics.

    Per-sample output dict:
        experiment_id, method, country_id, target_region_id, active_region_ids,
        split_role, K, seed, variable, metric, value, n_valid_pixels,
        n_time_steps, protocol_freeze_id
    """
    results: List[Dict] = []
    n_time_steps = len(dataset)

    for idx in range(len(dataset)):
        sample = dataset[idx]
        pred = predictor.predict(sample)

        for variable in ("surface", "rootzone"):
            true_analysis = sample[f"analysis_{variable}"]
            pred_analysis = pred[f"pred_analysis_{variable}"]
            true_inc = sample[f"increment_{variable}"]
            pred_inc = pred[f"pred_increment_{variable}"]
            forecast = sample[f"forecast_{variable}"]
            mask = sample["metric_mask"]

            n_valid = valid_pixel_count(mask)

            # analysis metrics
            a_rmse = analysis_rmse(pred_analysis, true_analysis, mask)
            a_mae = analysis_mae(pred_analysis, true_analysis, mask)
            a_skill = analysis_skill_vs_forecast(pred_analysis, true_analysis, forecast, mask)

            # increment metrics
            i_rmse = increment_rmse(pred_inc, true_inc, mask)
            i_mae = increment_mae(pred_inc, true_inc, mask)
            i_bias = increment_bias(pred_inc, true_inc, mask)
            i_corr = increment_corr(pred_inc, true_inc, mask)
            i_sign = sign_accuracy_deadzone(pred_inc, true_inc, mask)

            metrics = {
                "analysis_rmse": a_rmse,
                "analysis_mae": a_mae,
                "analysis_skill_vs_forecast": a_skill,
                "increment_rmse": i_rmse,
                "increment_mae": i_mae,
                "increment_bias": i_bias,
                "increment_corr": i_corr,
                "sign_accuracy_deadzone": i_sign,
            }

            for metric_name, value in metrics.items():
                results.append({
                    "experiment_id": experiment_id,
                    "method": predictor.method_id,
                    "country_id": sample["country_id"],
                    "target_region_id": sample["target_region_id"],
                    "active_region_ids": sample["active_region_ids"],
                    "split_role": split_role,
                    "K": sample["K"],
                    "seed": sample["seed"],
                    "variable": variable,
                    "metric": metric_name,
                    "value": value,
                    "n_valid_pixels": n_valid,
                    "n_time_steps": n_time_steps,
                    "protocol_freeze_id": protocol_freeze_id,
                })

    return results
```

- [ ] **Step 2: Verify it imports cleanly**

Run: `python -c "from hydroda.evaluation import evaluate_split; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add hydroda/evaluation/__init__.py hydroda/evaluation/harness.py
git commit -m "feat(phase3A): add evaluation harness"
```

---

## Task 4: `scripts/run_phase3A.py`

**Files:**
- Create: `scripts/run_phase3A.py`

- [ ] **Step 1: Write the script**

```python
#!/usr/bin/env python
"""Phase 3A — run forecast-only baseline for all 240 US LORO splits."""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hydroda.data.dataset import HydroDADataset
from hydroda.baselines.forecast import ForecastBaseline
from hydroda.evaluation.harness import evaluate_split

ROOT = Path(__file__).parent.parent
ARTIFACTS = ROOT / "artifacts"
METRICS_DIR = ARTIFACTS / "metrics" / "phase3A_forecast_only_US"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

DA_NC = "/fastersharefiles2/fenglonghan/dataset/SMAP/DA.nc"
REGION_MASKS_NC = str(ARTIFACTS / "regions" / "US_region_masks.nc")
SPLITS_JSON = str(ARTIFACTS / "splits" / "US_loro_kdate_splits.json")
FREEZE_MANIFEST = str(ARTIFACTS / "protocol" / "US_region_split_freeze_manifest.json")

with open(FREEZE_MANIFEST) as f:
    freeze = json.load(f)
PROTOCOL_FREEZE_ID = freeze["freeze_id"]

REGIONS = ["US-R1", "US-R2", "US-R3", "US-R4", "US-R5", "US-R6"]
K_VALUES = [0, 4, 12, 24]
SEEDS = list(range(10))
SPLIT_ROLES = ["source_train", "target_support", "target_query"]

EXPERIMENT_ID = "phase3A_forecast_only_US"
METHOD = ForecastBaseline()


def main():
    all_results = []

    for region in REGIONS:
        for K in K_VALUES:
            for seed in SEEDS:
                for split_role in SPLIT_ROLES:
                    dataset = HydroDADataset(
                        da_nc_path=DA_NC,
                        region_masks_nc=REGION_MASKS_NC,
                        splits_json=SPLITS_JSON,
                        target_region=region,
                        split_type=split_role,
                        K=K,
                        seed=seed,
                        freeze_manifest=FREEZE_MANIFEST,
                    )
                    results = evaluate_split(
                        dataset=dataset,
                        predictor=METHOD,
                        split_role=split_role,
                        experiment_id=EXPERIMENT_ID,
                        protocol_freeze_id=PROTOCOL_FREEZE_ID,
                    )
                    all_results.extend(results)
                    dataset.close()

    # Build DataFrame
    df = pd.DataFrame(all_results)
    csv_path = METRICS_DIR / "metrics_long.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {len(df)} rows to {csv_path}")
    print(f"Regions: {df['target_region_id'].unique().tolist()}")
    print(f"K values: {df['K'].unique().tolist()}")
    print(f"Split roles: {df['split_role'].unique().tolist()}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test with one split**

```bash
cd /sharefiles1/fenglonghan/projects/hydroda_ood
python -c "
import sys; sys.path.insert(0, '.')
from hydroda.data.dataset import HydroDADataset
from hydroda.baselines.forecast import ForecastBaseline
from hydroda.evaluation.harness import evaluate_split
import json

with open('artifacts/protocol/US_region_split_freeze_manifest.json') as f:
    freeze = json.load(f)

ds = HydroDADataset(
    da_nc_path='/fastersharefiles2/fenglonghan/dataset/SMAP/DA.nc',
    region_masks_nc='artifacts/regions/US_region_masks.nc',
    splits_json='artifacts/splits/US_loro_kdate_splits.json',
    target_region='US-R1',
    split_type='target_query',
    K=0,
    seed=0,
    freeze_manifest='artifacts/protocol/US_region_split_freeze_manifest.json',
)
baseline = ForecastBaseline()
results = evaluate_split(ds, baseline, 'target_query', 'phase3A_forecast_only_US', freeze['freeze_id'])
print(f'Got {len(results)} result rows for one split')
ds.close()
"
```

Expected: prints row count with no exceptions

- [ ] **Step 3: Commit**

```bash
git add scripts/run_phase3A.py
git commit -m "feat(phase3A): add run_phase3A.py entry point"
```

---

## Task 5: `tests/test_forecast_baseline.py` — remaining tests

**Files:**
- Modify: `tests/test_forecast_baseline.py`

Add these tests after the first 3 already written in Task 1:

```python
def test_metrics_long_schema():
    """CSV columns match spec exactly."""
    import pandas as pd
    from pathlib import Path
    csv_path = Path("artifacts/metrics/phase3A_forecast_only_US/metrics_long.csv")
    if not csv_path.exists():
        pytest.skip("CSV not yet generated")
    df = pd.read_csv(csv_path)
    expected_cols = [
        "experiment_id", "method", "country_id", "target_region_id",
        "active_region_ids", "split_role", "K", "seed", "variable",
        "metric", "value", "n_valid_pixels", "n_time_steps", "protocol_freeze_id",
    ]
    assert df.columns.tolist() == expected_cols, f"Got {df.columns.tolist()}"


def test_no_target_query_training_in_baseline():
    """ForecastBaseline uses no target_query data for fitting."""
    baseline = ForecastBaseline()
    assert baseline.method_id == "forecast_only"
    # Verify no training whatsoever
    assert not hasattr(baseline, "_fitted")
    # Calling predict multiple times doesn't change anything
    sample = {
        "forecast_surface": np.ones((10, 10), dtype=np.float32),
        "forecast_rootzone": np.ones((10, 10), dtype=np.float32),
    }
    r1 = baseline.predict(sample)
    r2 = baseline.predict(sample)
    np.testing.assert_array_equal(r1["pred_increment_surface"], r2["pred_increment_surface"])


def test_region_balanced_aggregation():
    """Per-region mean weighted equally regardless of pixel count."""
    import pandas as pd
    csv_path = Path("artifacts/metrics/phase3A_forecast_only_US/metrics_long.csv")
    if not csv_path.exists():
        pytest.skip("CSV not yet generated")
    df = pd.read_csv(csv_path)
    # Group by target_region_id and take mean of analysis_rmse
    region_means = df[df["metric"] == "analysis_rmse"].groupby("target_region_id")["value"].mean()
    # All 6 regions should be present
    assert len(region_means) == 6, f"Expected 6 regions, got {len(region_means)}"
    # Each mean should be finite
    assert all(np.isfinite(v) for v in region_means.values), f"Non-finite values: {region_means}"
```

- [ ] **Run all tests**

Run: `python -m pytest tests/test_forecast_baseline.py -v`
Expected: All 7 tests pass

- [ ] **Commit**

```bash
git add tests/test_forecast_baseline.py
git commit -m "feat(phase3A): add all 7 sanity tests for forecast baseline"
```

---

## Task 6: Report `reports/experiments/phase3A_forecast_only_US.md`

**Files:**
- Create: `reports/experiments/phase3A_forecast_only_US.md`

- [ ] **Step 1: Write report generation script (run after metrics CSV exists)**

```python
#!/usr/bin/env python
"""Generate Phase 3A forecast-only report."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
METRICS_CSV = ROOT / "artifacts/metrics/phase3A_forecast_only_US/metrics_long.csv"
STATS_JSON = ROOT / "artifacts/regions/US_region_stats.json"
FREEZE_MANIFEST = ROOT / "artifacts/protocol/US_region_split_freeze_manifest.json"
REPORT_PATH = ROOT / "reports/experiments/phase3A_forecast_only_US.md"

with open(FREEZE_MANIFEST) as f:
    freeze = json.load(f)
with open(STATS_JSON) as f:
    region_stats = json.load(f)

df = pd.read_csv(METRICS_CSV)
QUERY_DF = df[df["split_role"] == "target_query"]

REGIONS = ["US-R1", "US-R2", "US-R3", "US-R4", "US-R5", "US-R6"]
VARIABLES = ["surface", "rootzone"]

lines = [
    "# Phase 3A — Forecast-Only Baseline (US)",
    f"**Experiment ID:** phase3A_forecast_only_US",
    f"**Protocol Freeze ID:** {freeze['freeze_id']}",
    f"**Generated:** 2026-05-06",
    "",
    "## No-Leakage Statement",
    "",
    "The forecast-only baseline uses **no training data**. Predicted increment = 0 for all pixels.",
    "No target_query labels, analysis increments, or model errors were used in model selection or normalization.",
    "Metric masking uses `base_valid_mask` + `active_region_mask` + finite check only.",
    "",
    "## Per-Region RMSE — Target Query (m³/m³)",
    "",
    "| Region | Regime | Surface RMSE | Rootzone RMSE | Valid Coverage % |",
    "|--------|--------|-------------|--------------|-----------------|",
]

for region in REGIONS:
    rstats = region_stats.get(region, {})
    regime = rstats.get("regime", "unknown")
    row_data = {"region": region, "regime": regime}
    for var in VARIABLES:
        var_df = QUERY_DF[
            (QUERY_DF["target_region_id"] == region) &
            (QUERY_DF["variable"] == var) &
            (QUERY_DF["metric"] == "analysis_rmse")
        ]
        row_data[f"{var}_rmse"] = var_df["value"].mean() if len(var_df) > 0 else np.nan

    # Valid coverage: mean effective_mask_fraction across samples
    mask_df = QUERY_DF[
        (QUERY_DF["target_region_id"] == region) &
        (QUERY_DF["metric"] == "analysis_rmse")
    ]
    # effective_mask_fraction is embedded via n_valid_pixels ratio — compute per-sample
    # For this report we approximate via the mean n_valid_pixels vs region pixel count
    total_pix = freeze["region_pixel_counts"].get(region, np.nan)
    if not np.isnan(total_pix):
        mean_valid = mask_df["n_valid_pixels"].mean() if len(mask_df) > 0 else 0
        coverage = 100 * mean_valid / total_pix
    else:
        coverage = np.nan

    row_data["coverage"] = coverage
    lines.append(
        f"| {row_data['region']} | {row_data['regime']} | "
        f"{row_data['surface_rmse']:.4f} | {row_data['rootzone_rmse']:.4f} | "
        f"{row_data['coverage']:.1f}% |"
    )

# True increment RMSE table
lines += ["", "## Per-Region True Increment RMSE — Target Query (m³/m³)", ""]
lines.append("| Region | Surface Incr RMSE | Rootzone Incr RMSE |")
lines.append("|--------|------------------|-------------------|")

for region in REGIONS:
    row_data = {"region": region}
    for var in VARIABLES:
        var_df = QUERY_DF[
            (QUERY_DF["target_region_id"] == region) &
            (QUERY_DF["variable"] == var) &
            (QUERY_DF["metric"] == "increment_rmse")
        ]
        row_data[f"{var}_rmse"] = var_df["value"].mean() if len(var_df) > 0 else np.nan
    lines.append(
        f"| {row_data['region']} | {row_data['surface_rmse']:.4f} | {row_data['rootzone_rmse']:.4f} |"
    )

# Sanity table
lines += ["", "## Forecast-Only Sanity Check", ""]
lines.append("| Check | Value | Status |")
lines.append("|-------|-------|--------|")

fcst_only_check = QUERY_DF[
    (QUERY_DF["metric"] == "pred_increment_surface") &
    (QUERY_DF["variable"] == "surface")
]
if len(fcst_only_check) > 0:
    mean_pred_inc = fcst_only_check["value"].mean()
    sanity_status = "PASS" if abs(mean_pred_inc) < 1e-6 else "FAIL"
else:
    mean_pred_inc = np.nan
    sanity_status = "N/A"

lines.append(f"| pred_increment_surface mean | {mean_pred_inc:.6f} | {sanity_status} |")

# analysis_skill should be ≈ 0
skill_df = QUERY_DF[
    (QUERY_DF["metric"] == "analysis_skill_vs_forecast") &
    (QUERY_DF["variable"] == "surface")
]
mean_skill = skill_df["value"].mean() if len(skill_df) > 0 else np.nan
skill_status = "PASS" if abs(mean_skill) < 0.05 else "CHECK"
lines.append(f"| analysis_skill_vs_forecast mean | {mean_skill:.4f} | {skill_status} |")

# Warnings
lines += ["", "## Warnings", ""]
for warning in freeze.get("warnings", []):
    lines.append(f"- **{warning['region']}**: {warning['issue']} — {warning['note']}")

lines += ["", "## Output Files", ""]
lines.append(f"- CSV: `artifacts/metrics/phase3A_forecast_only_US/metrics_long.csv`")
lines.append(f"- Rows: {len(df)}")

REPORT_PATH.write_text("\n".join(lines))
print(f"Wrote report to {REPORT_PATH}")
```

- [ ] **Step 2: Run report script**

```bash
cd /sharefiles1/fenglonghan/projects/hydroda_ood
python scripts/run_phase3A.py && python scripts/generate_phase3A_report.py
```

- [ ] **Step 3: Commit**

```bash
git add reports/experiments/phase3A_forecast_only_US.md
git commit -m "feat(phase3A): add forecast-only baseline report"
```

---

## Task 7: Final verification

- [ ] Run: `python -m pytest tests/test_forecast_baseline.py -v`
- [ ] Run: `python -c "import pandas as pd; df = pd.read_csv('artifacts/metrics/phase3A_forecast_only_US/metrics_long.csv'); print(df.columns.tolist(), df.shape)"`
- [ ] Verify all 240 splits were processed (6 regions × 4 K values × 10 seeds × 3 split roles = 720 dataset evaluations, but each evaluation has multiple rows per sample)
- [ ] Final commit if needed
