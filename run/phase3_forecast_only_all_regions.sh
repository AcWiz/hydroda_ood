#!/bin/bash
# Phase 3 Forecast-only baseline: all 6 US regions × target_eval + source_test
# No model, no training, no GPU — pure data computation.
# Output: per-region metrics_long.csv + summary.json + overall summary table
set -euo pipefail

cd "$(dirname "$0")/.."

REGIONS=("US-R1" "US-R2" "US-R3" "US-R4" "US-R5" "US-R6")
OUTPUT_BASE="artifacts/results/phase3_forecast_only_all_regions"
SPLIT_TYPES=("target_eval" "source_test")

mkdir -p "${OUTPUT_BASE}"

echo "================================================================"
echo "Phase 3 Forecast-Only Baseline — All Regions"
echo "Output base: ${OUTPUT_BASE}"
echo "Regions: ${REGIONS[*]}"
echo "Split types: ${SPLIT_TYPES[*]}"
echo "Started at $(date)"
echo "================================================================"

for split_type in "${SPLIT_TYPES[@]}"; do
    echo ""
    echo "--- split_type=${split_type} ---"
    for region in "${REGIONS[@]}"; do
        echo ""
        echo "[${region}] ${split_type} ..."
        PYTHONPATH=. python scripts/eval/forecast_only_target_eval.py \
            --target_region "${region}" \
            --split_type "${split_type}" \
            --output_dir "${OUTPUT_BASE}/${split_type}"
        echo "[${region}] ${split_type} done."
    done
done

echo ""
echo "================================================================"
echo "All evaluations complete at $(date)"
echo "================================================================"

# Generate summary table
echo ""
echo "Generating summary table..."
PYTHONPATH=. python -c "
import json
from pathlib import Path

BASE = Path('${OUTPUT_BASE}')
SPLIT_TYPES = ['target_eval', 'source_test']
REGIONS = ['US-R1', 'US-R2', 'US-R3', 'US-R4', 'US-R5', 'US-R6']

rows = []
for st in SPLIT_TYPES:
    for r in REGIONS:
        sj = BASE / st / r / 'summary.json'
        if not sj.exists():
            rows.append({'split': st, 'region': r, 'status': 'MISSING'})
            continue
        s = json.loads(sj.read_text())
        surf = s.get('surface', {})
        rz   = s.get('rootzone', {})
        # Region label: for source_test, show active (evaluated) regions
        if st == 'source_test':
            active_regions = s.get('regions', [r])
            region_label = '|'.join(x.replace('US-', '') for x in active_regions)
        else:
            region_label = r
        rows.append({
            'split': st,
            'region': region_label,
            'n_dates': s.get('n_dates', 0),
            'n_pixels': s.get('n_valid_pixels_total', 0),
            'surf_analysis_rmse': surf.get('analysis_rmse_latw_mean'),
            'surf_increment_rmse': surf.get('increment_rmse_latw_mean'),
            'surf_skill_global': surf.get('analysis_skill_vs_forecast_global'),
            'rz_analysis_rmse': rz.get('analysis_rmse_latw_mean'),
            'rz_increment_rmse': rz.get('increment_rmse_latw_mean'),
            'rz_skill_global': rz.get('analysis_skill_vs_forecast_global'),
        })

# Print markdown table
print()
print('## Forecast-Only Baseline — All Regions Summary')
print()
print('| Split | Region | N_Dates | N_Pixels | Surf WRMSE | Surf IncWRMSE | Surf Skill | RZ WRMSE | RZ IncWRMSE | RZ Skill |')
print('|-------|--------|---------|----------|-----------|-------------|------------|---------|------------|----------|')
for r in rows:
    def f(v):
        if isinstance(v, float):
            return f'{v:.10f}'
        if isinstance(v, int):
            return f'{v:,}'
        return str(v)
    print(f'| {r[\"split\"]} | {r[\"region\"]} | {f(r[\"n_dates\"])} | {f(r[\"n_pixels\"])} | {f(r[\"surf_analysis_rmse\"])} | {f(r[\"surf_increment_rmse\"])} | {f(r[\"surf_skill_global\"])} | {f(r[\"rz_analysis_rmse\"])} | {f(r[\"rz_increment_rmse\"])} | {f(r[\"rz_skill_global\"])} |')

# Save markdown to file
md_path = BASE / 'summary_table.md'
lines = []
lines.append('# Forecast-Only Baseline — All Regions Summary')
lines.append('')
lines.append(f'Generated: {__import__(\"datetime\").datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}')
lines.append('')
lines.append('| Split | Region | N_Dates | N_Pixels | Surf WRMSE | Surf IncWRMSE | Surf Skill | RZ WRMSE | RZ IncWRMSE | RZ Skill |')
lines.append('|-------|--------|---------|----------|-----------|-------------|------------|---------|------------|----------|')
for r in rows:
    def f2(v):
        if isinstance(v, float):
            return f'{v:.10f}'
        if isinstance(v, int):
            return f'{v:,}'
        return str(v)
    lines.append(f'| {r[\"split\"]} | {r[\"region\"]} | {f2(r[\"n_dates\"])} | {f2(r[\"n_pixels\"])} | {f2(r[\"surf_analysis_rmse\"])} | {f2(r[\"surf_increment_rmse\"])} | {f2(r[\"surf_skill_global\"])} | {f2(r[\"rz_analysis_rmse\"])} | {f2(r[\"rz_increment_rmse\"])} | {f2(r[\"rz_skill_global\"])} |')
md_path.write_text('\n'.join(lines) + '\n')
print()
print(f'Summary table saved to {md_path}')
"

echo ""
echo "Done. Output: ${OUTPUT_BASE}/"
