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
SPLITS_JSON = 'artifacts/splits/US_loro_kdate_splits.json'
MANIFEST = 'artifacts/protocol/US_region_split_freeze_manifest.json'

region = '${TARGET_REGION}'
print(f'Running forecast-only evaluation for {region}...')

ds = HydroDADataset(
    da_nc_path=f'{DATA_DIR}/DA.nc',
    region_masks_nc=REGION_MASKS,
    splits_json=SPLITS_JSON,
    target_region=region,
    split_type='target_query',
    K=4, seed=0,
    freeze_manifest=MANIFEST,
)

predictor = ForecastBaseline()
rows = evaluate_split(
    dataset=ds,
    predictor=predictor,
    split_role='target_query',
    experiment_id=f'phase3A_{region}',
    protocol_freeze_id='hyperda_v4_final_2015_2025_context2022_query2023_2025_k0_4_12',
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
surface_rmse = df[(df['variable']=='surface') & (df['metric']=='increment_rmse')]['value'].mean()
rootzone_rmse = df[(df['variable']=='rootzone') & (df['metric']=='increment_rmse')]['value'].mean()
surface_skill = df[(df['variable']=='surface') & (df['metric']=='analysis_skill_vs_forecast')]['value'].mean()
rootzone_skill = df[(df['variable']=='rootzone') & (df['metric']=='analysis_skill_vs_forecast')]['value'].mean()
surface_corr = df[(df['variable']=='surface') & (df['metric']=='analysis_correlation')]['value'].mean()
rootzone_corr = df[(df['variable']=='rootzone') & (df['metric']=='analysis_correlation')]['value'].mean()

print()
print('=' * 60)
print('Phase 3 Forecast-only Results Summary')
print('=' * 60)
print(f'  Region: {region}')
print(f'  Split:  target_query')
print()
print(f'  Surface:    RMSE={surface_rmse:.6f}  Skill={surface_skill:.6f}  Corr={surface_corr:.4f}')
print(f'  Rootzone:  RMSE={rootzone_rmse:.6f}  Skill={rootzone_skill:.6f}  Corr={rootzone_corr:.4f}')
print()
print(f'  Total rows: {len(rows)}')
print('=' * 60)
"