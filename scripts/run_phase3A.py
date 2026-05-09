#!/usr/bin/env python3
"""Phase 3A — Forecast-only baseline evaluation harness.

Evaluates target_query for all 240 splits (6 regions × 4 K × 10 seeds).
target_query is the most important split for evaluation as it covers 2022-2025.

Output: artifacts/metrics/phase3A_forecast_only_US/metrics_long.csv
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import netCDF4

from hydroda.baselines.forecast import ForecastBaseline
from hydroda.evaluation.harness import evaluate_split


DA_NC = "/fastersharefiles2/fenglonghan/dataset/SMAP/DA.nc"
REGION_MASKS_NC = "artifacts/regions/US_region_masks.nc"
SPLITS_JSON = "artifacts/splits/US_loro_kdate_splits.json"
FREEZE_MANIFEST = "artifacts/protocol/US_region_split_freeze_manifest.json"
OUTPUT_DIR = Path("artifacts/metrics/phase3A_forecast_only_US")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(FREEZE_MANIFEST) as f:
    FREEZE_ID = json.load(f)["freeze_id"]

EXPERIMENT_ID = "phase3A_forecast_only_US"
REGIONS = ["US-R1", "US-R2", "US-R3", "US-R4", "US-R5", "US-R6"]
K_VALUES = [0, 4, 12, 24]
SEEDS = list(range(10))
SPLIT_ROLES = ["source_train", "target_support", "target_query"]

_ALL_REGIONS = ["US-R1", "US-R2", "US-R3", "US-R4", "US-R5", "US-R6"]
NC_CHUNK = 200


class FastDataset:
    def __init__(self, samples: list, K: int, seed: int):
        self._samples = samples
        self.K = K
        self.seed = seed

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        return self._samples[idx]

    def close(self) -> None:
        pass


def load_data_chunked(nc_path: str, time_indices: list, chunk: int = NC_CHUNK):
    """Load input and target arrays in small chunks."""
    ds = netCDF4.Dataset(nc_path, "r")
    input_var = ds.variables["input"]
    target_var = ds.variables["target"]

    all_input = []
    all_target = []
    t_arr = np.array(time_indices)
    for start in range(0, len(t_arr), chunk):
        end = min(start + chunk, len(t_arr))
        inp = input_var[t_arr[start:end]].astype(np.float32)
        tgt = target_var[t_arr[start:end]].astype(np.float32)
        all_input.append(inp)
        all_target.append(tgt)
        del inp, tgt

    input_all = np.concatenate(all_input, axis=0)
    target_all = np.concatenate(all_target, axis=0)
    ds.close()
    return input_all, target_all


def run_phase3A() -> pd.DataFrame:
    """Run forecast-only on target_query for all 240 splits."""
    predictor = ForecastBaseline()

    with open(SPLITS_JSON) as f:
        splits_data = json.load(f)
    splits_by_key = {
        (s["target_region_id"], s["K"], s["seed"]): s
        for s in splits_data["splits"]
    }

    with open("artifacts/regions/US_region_stats.json") as f:
        region_stats = json.load(f)

    import xarray as xr
    region_ds = xr.open_dataset(REGION_MASKS_NC)
    region_mask_int = region_ds["region_mask_integer"].values.astype(np.int16)
    region_ds.close()

    all_results = []
    total_evals = len(REGIONS) * len(K_VALUES) * len(SEEDS)
    print(f"Phase 3A: {total_evals} target_query evaluations")

    t_start = time.time()

    for target_region in REGIONS:
        for K in K_VALUES:
            # Pre-load target_query data for all 10 seeds at once
            split_entry = splits_by_key[(target_region, K, 0)]
            regime_id = region_stats[target_region]["regime"]
            active_region_ids = [r for r in _ALL_REGIONS if r != target_region]
            rnum_list = [int(rid.split("-R")[1]) for rid in active_region_ids]
            active_region_mask = np.isin(region_mask_int, rnum_list).astype(np.float32)
            target_rnum = int(target_region.split("-R")[1])
            target_eval_mask = np.isin(region_mask_int, [target_rnum]).astype(np.float32)

            date_list = split_entry["target_query_dates"]
            time_indices = [d["time_index"] for d in date_list]
            date_str_map = {d["time_index"]: d["date_str"] for d in date_list}

            t_load = time.time()
            input_all, target_all = load_data_chunked(DA_NC, time_indices, NC_CHUNK)
            load_time = time.time() - t_load
            print(f"  {target_region} K={K}: {len(time_indices)} samples loaded in {load_time:.1f}s")

            # Build samples for each seed (same data, different seed/K in metadata)
            for seed in SEEDS:
                samples = []
                for i, (ti, inp, tgt) in enumerate(zip(time_indices, input_all, target_all)):
                    forecast_s = inp[0]
                    forecast_r = inp[1]
                    analysis_s = tgt[0]
                    analysis_r = tgt[1]
                    inc_s = analysis_s - forecast_s
                    inc_r = analysis_r - forecast_r
                    base_mask = (inp[11] > 0.5).astype(np.float32)
                    loss_mask = (
                        active_region_mask.astype(bool)
                        & (base_mask > 0.5)
                        & np.isfinite(forecast_s)
                        & np.isfinite(forecast_r)
                        & np.isfinite(analysis_s)
                        & np.isfinite(analysis_r)
                    ).astype(np.float32)

                    samples.append({
                        "forecast_surface": forecast_s,
                        "forecast_rootzone": forecast_r,
                        "analysis_surface": analysis_s,
                        "analysis_rootzone": analysis_r,
                        "increment_surface": inc_s,
                        "increment_rootzone": inc_r,
                        "base_valid_mask": base_mask,
                        "region_mask_integer": region_mask_int,
                        "active_region_mask": active_region_mask,
                        "loss_mask": active_region_mask,
                        "metric_mask": (
                            target_eval_mask
                            & (base_mask > 0.5)
                            & np.isfinite(forecast_s)
                            & np.isfinite(forecast_r)
                            & np.isfinite(analysis_s)
                            & np.isfinite(analysis_r)
                        ).astype(np.float32),
                        "date_str": date_str_map.get(ti, "unknown"),
                        "time_index": int(ti),
                        "country_id": "US",
                        "target_region_id": target_region,
                        "active_region_ids": active_region_ids,
                        "split_role": "target_query",
                        "regime_id": regime_id,
                        "split_id": f"{target_region}-K{K}-S{seed}-target_query",
                        "K": K,
                        "seed": seed,
                    })

                dataset = FastDataset(samples, K=K, seed=seed)
                experiment_id = f"{EXPERIMENT_ID}_{target_region}_K{K}_S{seed}_target_query"

                results = evaluate_split(
                    dataset=dataset,
                    predictor=predictor,
                    split_role="target_query",
                    experiment_id=experiment_id,
                    protocol_freeze_id=FREEZE_ID,
                    preloaded=False,
                )

                for r in results:
                    r["method"] = "forecast_only"
                all_results.extend(results)

                elapsed = time.time() - t_start
                eval_count = sum(1 for _ in range(len(SEEDS)))
                count = (REGIONS.index(target_region) * len(K_VALUES) + K_VALUES.index(K)) * len(SEEDS) + seed + 1
                rate = count / elapsed if elapsed > 0 else 0
                eta = (total_evals - count) / rate if rate > 0 else 0
                if count % 50 == 0 or count == total_evals:
                    print(f"  [{count:3d}/{total_evals}] "
                          f"elapsed={elapsed:.0f}s ETA={eta:.0f}s")

    elapsed_total = time.time() - t_start
    print(f"\n  Done: {len(all_results)} rows in {elapsed_total:.0f}s")

    df = pd.DataFrame(all_results)
    out_path = OUTPUT_DIR / "metrics_long.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")

    return df


if __name__ == "__main__":
    df = run_phase3A()
    print(f"\nShape: {df.shape}")
    print(df["metric"].value_counts())
