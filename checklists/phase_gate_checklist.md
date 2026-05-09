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
- [ ] K-date split manifests 生成。
- [ ] no-leakage split tests 通过。

## Phase 3 -> Phase 4

- [ ] Forecast baseline 通过 sanity checks。
- [ ] source mean / target mean / monthly / ridge 已跑通。
- [ ] metrics_long.csv 生成。
- [ ] simple baseline table 可生成。

## Phase 4 -> Phase 5

- [ ] neural source-only baseline 可复现。
- [ ] tiny batch overfit 通过。
- [ ] no query label early stopping。

## Phase 5 -> Phase 6

- [ ] HyRAO ablation 完成。
- [ ] trainable parameter logs 存在。
- [ ] sparse block selection logs 存在。
