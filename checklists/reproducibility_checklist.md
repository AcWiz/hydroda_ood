# Reproducibility Checklist

- [ ] 每个 experiment 有唯一 experiment_id。
- [ ] 每个 run 保存 config。
- [ ] 每个 run 保存 split manifest path。
- [ ] 每个 run 保存 seed。
- [ ] 每个 run 保存 package versions。
- [ ] 每个 run 保存 git commit hash，如果可用。
- [ ] metrics 使用 long-form CSV。
- [ ] tables 可由脚本重新生成。
- [ ] failed runs 被显式记录。
- [ ] 随机 support seeds 固定且可复现。
