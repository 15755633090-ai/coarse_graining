# Lipo seed 0：修正执行协议后的结果

仅汇总保存的最佳 checkpoint，不重新评估测试集。轮次从 1 开始。

| 实验 | 总轮数 | 最佳轮次 | 验证 RMSE ↓ | 测试 RMSE ↓ | 测试 MAE ↓ | 测试 R² ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Stage-1 Base | 107 | 77 | 0.67109 | 0.68153 | 0.50975 | 0.61524 |
| Stage-2 Base | 37 | 7 | 0.67518 | 0.69381 | 0.52126 | 0.60126 |
| Stage-2 Multiscale | 56 | 26 | 0.81584 | 0.79306 | 0.63383 | 0.47902 |

粗粒化相对 Base2：测试 RMSE 增加 0.09925（14.31%），验证 RMSE 增加 0.14066。

本次 seed 0 下，当前粗粒化实现未带来性能收益。该结果不构成跨随机种子的统计结论，也不能单凭指标判定具体退化机制。

Stage-1 的全部 107 轮验证轨迹复现旧 Base。两组 Stage-2 使用相同来源 encoder、数据划分、scaler、随机种子和训练超参数；best/last encoder 权重与来源逐元素一致。两组读出结构不同，因此结果比较的是完整粗粒化方案与 Base2，而非隔离单一模块的因果效应。

## 核对

- PASS: Stage-1 Base best_epoch_matches_history
- PASS: Stage-1 Base completed_with_test
- PASS: Stage-2 Base best_epoch_matches_history
- PASS: Stage-2 Base completed_with_test
- PASS: Stage-2 Base source_hash_matches
- PASS: Stage-2 Base encoder_best_and_last_unchanged
- PASS: Stage-2 Base scaler_matches_source
- PASS: Stage-2 Multiscale best_epoch_matches_history
- PASS: Stage-2 Multiscale completed_with_test
- PASS: Stage-2 Multiscale source_hash_matches
- PASS: Stage-2 Multiscale encoder_best_and_last_unchanged
- PASS: Stage-2 Multiscale scaler_matches_source
- PASS: paired_training_protocol_equal
- PASS: paired_seeds_equal
- PASS: dataset_and_split_equal
- PASS: stage1_all_validation_epochs_reproduce_reference
