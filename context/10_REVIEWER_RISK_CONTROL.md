# 10 Reviewer Risk Control

## 1. 高风险问题与防御

### R1. “这只是 soil moisture prediction 吗？”

防御：任务明确是 reference DA analysis-increment emulation，不声称自然真值。

### R2. “目标域标签预算是否真的受限？”

防御：主协议是 zero/few-shot；K=0 不使用 target labels，K=4/12 只使用 K 个
labeled target DA cycles 更新轻量变量。Target 2015-2021 input-side context 可用于
prompt，2023-2025 target_eval 严格 held out。

### R3. “是否有 query leakage？”

防御：split manifest + target_train/eval date hashes + no-leakage tests + normalization provenance。

### R4. “简单校正是否已经足够？”

防御：Forecast、source-only、prompt-conditioned shared、adapter/LoRA 作为主 baseline；
target-train mean/monthly/ridge 只作为 internal sanity 或附录。

### R5. “区域是否 cherry-pick？”

防御：区域在训练前固定，只使用 input/static/geophysical 信息，不使用 increment 或 model error。

### R6. “HyperDA-SAFE 是否只是 adapter/LoRA 换名？”

防御：必须展示 source-trained HyperDA prior、target_context input-only
monthly prompt prototypes、parameter-space lightweight operator generation 和
source-anchored K-shot refinement。HyRAO 是旧命名/旧方案，不作为主方法回应审稿。

---

## 2. 必须保留的 reviewer-proof artifacts

```text
audit reports
region quality reports
zero/few-shot split manifests
normalization provenance logs
support date lists
compact summaries / overview tables
per-region result tables
seed variance tables
```

没有这些 artifacts，不要声称 paper-ready。

---

## 3. 主文默认叙事

```text
1. Define the deployment problem.
2. Introduce HydroDA-OOD target_context/target_support/target_eval protocol.
3. Show forecast/source-only OOD gap.
4. Compare HyperDA K=0/4/12 zero/few-shot generalization.
5. Introduce HyperDA parameter generation.
6. Analyze lightweight adaptation ablations and high-update events after model selection.
```
