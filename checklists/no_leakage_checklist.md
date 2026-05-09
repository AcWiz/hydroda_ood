# No-Leakage Checklist

每次涉及数据、split、normalization、training、evaluation，必须逐项检查。

## Split leakage

- [ ] support dates 只来自 target support year。
- [ ] query dates 只来自 target query years。
- [ ] support 和 query 无重叠。
- [ ] support selection 未使用 analysis increment。
- [ ] support selection 未使用 model error。
- [ ] 所有方法在同一 region/K/seed 下使用相同 support dates。

## Normalization leakage

- [ ] input normalization stats 未使用 target query。
- [ ] target/increment normalization stats 未使用 target query labels。
- [ ] normalization provenance 被保存。

## Region leakage

- [ ] region 定义未使用 analysis increment。
- [ ] region 定义未使用模型结果。
- [ ] region 定义在训练前固定。

## Training leakage

- [ ] early stopping 未使用 target query labels。
- [ ] model selection 未使用 target query labels。
- [ ] K=0 未使用 target support labels。

## Evaluation leakage

- [ ] high-update mask 只用于 evaluation。
- [ ] event analysis 不用于重新选模型。
- [ ] metrics 使用 loss_mask，而非 obs_mask 单独替代。
