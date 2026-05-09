# 10 Reviewer Risk Control

## 1. 高风险问题与防御

### R1. “这只是 soil moisture prediction 吗？”

防御：任务明确是 reference DA analysis-increment emulation，不声称自然真值。

### R2. “few-shot 是否只是很多 patch？”

防御：K 是 DA dates/cycles；报告 effective support budget，但不把 patch 数当 K。

### R3. “是否有 query leakage？”

防御：split manifest + no-leakage tests + normalization provenance。

### R4. “简单校正是否已经足够？”

防御：Forecast、source mean、target mean、monthly mean、ridge 都作为主 baseline。

### R5. “区域是否 cherry-pick？”

防御：区域在训练前固定，只使用 input/static/geophysical 信息，不使用 increment 或 model error。

### R6. “HyRAO 是否只是 adapter/LoRA 换名？”

防御：必须展示 input-only descriptor、region latent、sparse support adaptation 的 ablation。

---

## 2. 必须保留的 reviewer-proof artifacts

```text
audit reports
region quality reports
split manifests
normalization provenance logs
support date lists
metric long-form csv
per-region result tables
seed variance tables
```

没有这些 artifacts，不要声称 paper-ready。

---

## 3. 主文默认叙事

```text
1. Define the deployment problem.
2. Introduce HydroDA-OOD and K-date protocol.
3. Show simple baselines and OOD gap.
4. Show adaptation difficulty under K=4/12.
5. Introduce HyRAO.
6. Analyze where HyRAO helps: high-update events, dryland, irrigation, vegetation opacity, mountain stress.
```
