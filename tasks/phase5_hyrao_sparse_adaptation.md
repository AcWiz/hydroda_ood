# Phase 5 — HyRAO and Sparse Adaptation

## 目标

实现论文主方法：Hydroclimate-conditioned Region-Adaptive Operator。

## 方法定义

```text
z_r = g_phi(d_r)
pred_increment = f_theta(x, z_r)
```

`d_r` 只能来自 input-only region descriptor。

## 需要实现

```text
hydroda/regions/descriptors.py
hydroda/models/hyrao.py
hydroda/adaptation/finetune.py
hydroda/adaptation/sparse_blocks.py
hydroda/adaptation/fisa.py
scripts/run_adaptation.py
tests/test_region_descriptor_no_labels.py
tests/test_adaptation_trainable_params.py
```

## Ablations

```text
source_only
full_finetune
adapter
LoRA
region_latent_only
region_latent_adapter
sparse_block_topk
HyRAO
HyRAO-FISA
```

## 验收标准

```text
1. descriptor 计算不读取 analysis/increment。
2. K=0 可运行。
3. K>0 adaptation 只使用 target support labels。
4. trainable parameter report 正确输出。
5. sparse block selection log 可复现。
```
