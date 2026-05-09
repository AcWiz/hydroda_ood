# 01 Research Contract

## 1. 论文级问题定义

HydroDA-OOD 研究真实部署问题：

> 当 neural land DA operator 被部署到新的水文气候区、新国家或新大陆时，目标区域通常只有少量历史 DA analysis cycles。我们研究如何在 K=0/4/12/24 target dates 下进行可靠、稀疏、高效的区域适配。

任务不是预测自然真值土壤湿度，而是模拟 reference DA analysis increment。

---

## 2. 最终论文贡献边界

主论文只保留三类核心贡献：

### C1. HydroDA-OOD benchmark

跨区域 / 跨大陆 neural land DA increment emulation benchmark。

阶段性版本：

```text
HydroDA-OOD-US: 6 US hydroclimatic regions, leave-one-region-out
```

完整版本：

```text
HydroDA-OOD-Full: US / CN / AU, 18 regions, cross-continent transfer
```

### C2. K-date few-cycle calibration protocol

K 表示 target DA analysis dates/cycles，不是 pixels/patches。

```text
K = 0, 4, 12, 24
```

### C3. HyRAO: hydroclimate-conditioned region-adaptive operator

方法不是普通 fine-tuning。HyRAO 必须包含：

```text
input-only region descriptor
region-conditioned modulation / latent
K-date sparse adaptation
```

---

## 3. 不应作为主贡献的内容

以下内容可以做 ablation / appendix，但不要宣传成主贡献：

```text
普通 LoRA
普通 adapter
普通 full fine-tuning
Hessian Top-K 本身
把 Sformer 用到 DA 数据
```

除非它们被证明是 HyRAO 的必要机制。

---

## 4. 审稿人视角下必须证明的命题

论文必须回答：

```text
Q1. Forecast-only baseline 到底有多强？
Q2. Source-only neural DA operator 是否存在 region OOD gap？
Q3. K-date support 是否比普通 source pooling 有稳定收益？
Q4. Target support mean / monthly mean / ridge 是否已经足够强？
Q5. HyRAO 是否在 K=4/12 这种低预算下优于 adapter / LoRA / full fine-tuning？
Q6. improvement 是否来自真实 increment learning，而不是 mask、seasonality 或均值偏差？
```

---

## 5. 必须保护的术语

使用：

```text
reference DA analysis
analysis increment
few-cycle target calibration
hydroclimatic shift
region-conditioned adaptation
input-only descriptor
query-label isolation
```

避免：

```text
ground truth soil moisture
true label
real soil moisture prediction
few-shot patches
```
