# Phase Gate Checklist

## Phase 0 -> Phase 1

- [ ] DA.nc dims/coords/data_vars 已报告。
- [ ] time coverage 已报告。
- [ ] variable/channel mapping 已报告或列为阻塞。
- [ ] mask semantics 初步确认。
- [ ] lat/lon availability 已确认。

## Phase 1 -> Phase 2

- [ ] HydroDADataset sample contract 通过测试。
- [ ] increment reconstruction 通过测试。
- [ ] mask keys 独立。
- [ ] normalization provenance 可追踪。

## Phase 2 -> Phase 3

- [ ] US-R1..US-R6 masks 生成。
- [ ] region quality report 生成。
- [ ] zero/few-shot split manifest `US_loro_zero_few_shot_splits.json` 生成。
- [ ] legacy target_train/K-date manifests 如保留，已标记为 reproduction only。
- [ ] no-leakage split tests 通过：target_context/support 与 target_eval 无重叠。

## Phase 3 -> Phase 4

- [ ] Forecast baseline 通过 sanity checks。
- [ ] source/support mean、monthly、ridge 等 internal sanity 明确不进入主表。
- [ ] compact summary/overview 可生成；长 `metrics_long.csv` 可再生。
- [ ] simple baseline table 使用 target_eval final-only metrics。

## Phase 4 -> Phase 5

- [ ] neural source-only baseline 可复现。
- [ ] tiny batch overfit 通过。
- [ ] no query label early stopping。

## Phase 5 -> Reporting

- [ ] HyperDA-SAFE K=0/4/12 在同一 zero/few-shot manifest 下完成。
- [ ] K=4/K=12 paper-facing runs 包含 `safe_policy.json` path/hash 和 `policy_source=source_side_episode_calibration`。
- [ ] `target_val_usage=unused_in_main_protocol`，target_eval 仅最终离线评估。
- [ ] trainable parameter logs、support loss、drift、anchor alpha 和 `adapt_mix_rho` metadata 存在。
- [ ] diagnostic/ablation wrappers 未混入 paper-main run surface。
