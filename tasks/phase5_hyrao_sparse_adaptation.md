# Phase 5 — Source Operator Episode Bank 与 HyperDA 准备

> 旧文件名保留为兼容 Claude command，但 V4 不再使用 HyRAO 作为主方法名。本阶段目标是 HyperDA 的 source operator episode bank。

## 目标

冻结 source-only backbone `θ0`，在 source episodes 上训练 lightweight operator parameters `ζ_d*`，构建 HyperDA 的 parameter-space supervision / prior。

## 必读

```text
context/01_RESEARCH_CONTRACT.md
specs/hyperda_v4.yaml
specs/protocol_v4.yaml
checklists/no_leakage_checklist.md
```

## Episode 定义

```text
d = continent × region × season × tile_group
```

US-only development 可用：

```text
source regions × seasons × tile groups
```

这些 episodes 不是 independent domains，论文中必须写成：

```text
structured local operator estimation problems used to learn a parameter-space prior
```

## 需要实现

```text
hydroda/models/adapters.py
hydroda/models/parameter_blocks.py
hydroda/operator_bank/train_episodes.py
hydroda/operator_bank/zeta_schema.py
hydroda/operator_bank/pack_zeta.py
hydroda/prompts/prompt_builder.py
scripts/run_operator_bank.py
tests/test_zeta_pack_unpack.py
tests/test_episode_no_query_access.py
```

## 每个 episode 保存

```text
operator_bank/{episode_id}/prompt.pt
operator_bank/{episode_id}/zeta.pt
operator_bank/{episode_id}/zeta_vector.pt
operator_bank/{episode_id}/zeta_schema.json
operator_bank/{episode_id}/train_metrics.json
operator_bank/{episode_id}/metadata.json
```

## 训练约束

```text
θ0 frozen
只训练 adapter / output-head residual / optional FiLM
不训练 full backbone
不读取 target query labels
```

## 验收标准

```text
1. zeta pack/unpack 完全可逆。
2. 每个 episode metadata 包含 region、season、tile_group、time range、valid pixel ratio。
3. episode 训练日志包含 loss curve 和 operator quality。
4. 没有任何 target query time 被用于 episode training 或 prompt building。
```
