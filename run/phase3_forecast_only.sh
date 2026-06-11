#!/bin/bash
# Phase 3 Forecast-only evaluation: US-R1 (regenerate metrics for Phase 3A report)
set -euo pipefail

TARGET_REGION="${1:-US-R1}"
export CUDA_VISIBLE_DEVICES="${2:-0}"
cd "$(dirname "$0")/.."

OUT_DIR="artifacts/metrics/phase3A_forecast_only_US"
mkdir -p "${OUT_DIR}"

PYTHONPATH=. python -c "
import sys
sys.path.insert(0, '.')
from scripts.eval.run_forecast_only_eval import run_forecast_only_fast, quick_diagnostic
from hydroda.baselines.forecast import ForecastBaseline
from hydroda.data.dataset import HydroDADataset
from hydroda.evaluation.harness import evaluate_split
from datetime import datetime

DATA_DIR = '/fastersharefiles2/fenglonghan/dataset/SMAP'
REGION_MASKS = 'artifacts/regions/US_region_masks.nc'
SPLITS_JSON = 'artifacts/splits/US_loro_zero_few_shot_splits.json'
MANIFEST = 'artifacts/protocol/US_region_split_freeze_manifest.json'

region = '${TARGET_REGION}'
print(f'Running forecast-only evaluation for {region}...')

ds = HydroDADataset(
    da_nc_path=f'{DATA_DIR}/DA.nc',
    region_masks_nc=REGION_MASKS,
    splits_json=SPLITS_JSON,
    target_region=region,
    split_type='target_eval',
    K=4, seed=0,
    freeze_manifest=MANIFEST,
)

predictor = ForecastBaseline()
rows = evaluate_split(
    dataset=ds,
    predictor=predictor,
    split_role='target_eval',
    experiment_id=f'phase3A_{region}',
    protocol_freeze_id='hyperda_v4_4_zero_few_shot_generalization_2015_2025_context2015_2021_sourceval2022_eval2023_2025',
    method='forecast_only',
    split_file=SPLITS_JSON,
    mask_file=REGION_MASKS,
    preloaded=False,
)
ds.close()

import pandas as pd
df = pd.DataFrame(rows)
df.to_csv('${OUT_DIR}/metrics_long.csv', index=False)
print(f'Saved {len(rows)} rows to ${OUT_DIR}/metrics_long.csv')

# Compute and print summary metrics
import numpy as np
surface_rmse = df[(df['variable']=='surface') & (df['metric']=='increment_rmse_latw')]['value'].mean()
rootzone_rmse = df[(df['variable']=='rootzone') & (df['metric']=='increment_rmse_latw')]['value'].mean()
surface_skill = df[(df['variable']=='surface') & (df['metric']=='analysis_skill_vs_forecast_latw')]['value'].mean()
rootzone_skill = df[(df['variable']=='rootzone') & (df['metric']=='analysis_skill_vs_forecast_latw')]['value'].mean()
surface_corr = df[(df['variable']=='surface') & (df['metric']=='increment_corr_latw')]['value'].mean()
rootzone_corr = df[(df['variable']=='rootzone') & (df['metric']=='increment_corr_latw')]['value'].mean()

print()
print('=' * 60)
print('Phase 3 Forecast-only Results Summary')
print('=' * 60)
print(f'  Region: {region}')
print(f'  Split:  target_eval')
print()
print(f'  Surface:    WRMSE={surface_rmse:.10f}  Skill_latw={surface_skill:.10f}  Corr_latw={surface_corr:.10f}')
print(f'  Rootzone:  WRMSE={rootzone_rmse:.10f}  Skill_latw={rootzone_skill:.10f}  Corr_latw={rootzone_corr:.10f}')
print()
print(f'  Total rows: {len(rows)}')
print('=' * 60)
"