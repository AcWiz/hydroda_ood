# 10 Reviewer Risk Control

## 1. 高风险问题与防御

### R1. “这只是 soil moisture prediction 吗？”

防御：任务明确是 reference DA analysis-increment emulation，不声称自然真值。

### R2. “目标域用了完整训练集，是否还叫泛化？”

防御：明确部署设定是 train/eval temporal holdout under target-domain shift：
2015-2021 target_train 可用于构造 target-specific operator，2023-2025 target_eval
严格 held out。旧 K-shot 只作为 secondary ablation。

### R3. “是否有 query leakage？”

防御：split manifest + target_train/eval date hashes + no-leakage tests + normalization provenance。

### R4. “简单校正是否已经足够？”

防御：Forecast、source-only、prompt-conditioned shared、adapter/LoRA 作为主 baseline；
target-train mean/monthly/ridge 只作为 internal sanity 或附录。

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
2. Introduce HydroDA-OOD target_train/target_eval protocol.
3. Show forecast/source-only OOD gap.
4. Compare target_full_train adaptation mechanisms.
5. Introduce HyperDA parameter generation.
6. Analyze legacy K-shot ablation and high-update events after model selection.
```
