# /run-hyperda-v4-smoke

运行 HyperDA V4 最小 smoke check。

必须检查：

```text
1. forecast-only pred_increment 是否全零。
2. metrics 是否把 analysis metrics 和 increment metrics 分开计算。
3. ProtocolConfig 时间段是否与 V4 一致。
4. LeakageGuard 是否拒绝 target_query labels 用于 prompt / normalization / model_selection。
5. SmallResUNet forward shape 是否为 [B, 2, H, W]。
6. HyperDA skeleton 是否只生成 lightweight zeta，不生成 full backbone。
```

建议命令：

```bash
python -m pytest tests/test_metrics.py tests/test_evaluation_harness_metric_routing.py -q
python -m pytest tests/test_protocol_leakage_guard.py -q
python -m pytest tests/test_neural_forward.py -q
```

如果真实数据不可访问，用 synthetic sample 运行，不要伪造真实实验结果。
