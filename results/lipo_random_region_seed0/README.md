# Lipo seed 0：随机区域模型结果汇总

从保存的 checkpoint 和训练日志汇总，未重新运行测试集。轮次从 1 开始。

| 模型 | 总轮数 | 最佳轮次 | 验证 RMSE ↓ | 测试 RMSE ↓ | 测试 MAE ↓ | 测试 R² ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Stage-1 Base | 107 | 77 | 0.67109 | 0.68153 | 0.50975 | 0.61524 |
| Stage-2 Base | 37 | 7 | 0.67518 | 0.69381 | 0.52126 | 0.60126 |
| 旧多尺度模型 | 56 | 26 | 0.81584 | 0.79306 | 0.63383 | 0.47902 |
| 随机区域（原版） | 37 | 7 | 0.66901 | 0.67590 | 0.51733 | 0.62157 |
| 随机区域（速度优化版） | 37 | 7 | 0.66901 | 0.67590 | 0.51733 | 0.62157 |

## 对比

- 相比 Stage-1 Base，测试 RMSE 下降 0.00563（0.83%）；测试 MAE 变化为 +0.00758。
- 相比 Stage-2 Base，测试 RMSE 下降 0.01790（2.58%）；测试 MAE 变化为 -0.00393。

## 解释与边界

- 新模型保留 H4 Sum/Mean 表示，与随机区域表示拼接后重新训练预测头；未使用冻结 Base predictor 加 residual correction。
- 配置：r=2，s=8，Kmax=8，单层四头距离 bias attention，训练 1 个视角、验证和测试 5 个固定视角。
- Stage-2 Base 与新模型使用同一 Stage-1 来源的冻结 encoder；Stage-1 Base 为微调阶段的较强参考。
- 优化版与原版是同一 seed 的实现复核，不能当作两个独立随机种子。当前结果不代表多 seed 稳定性。
- 训练日志没有可靠的逐轮耗时记录，本次汇总不推算整轮加速比例。

## 核验

- PASS: Stage-1 Base history_matches_checkpoint
- PASS: Stage-1 Base selected_best_epoch
- PASS: Stage-1 Base completed_with_test
- PASS: Stage-2 Base history_matches_checkpoint
- PASS: Stage-2 Base selected_best_epoch
- PASS: Stage-2 Base completed_with_test
- PASS: 旧多尺度模型 history_matches_checkpoint
- PASS: 旧多尺度模型 selected_best_epoch
- PASS: 旧多尺度模型 completed_with_test
- PASS: 随机区域（原版） history_matches_checkpoint
- PASS: 随机区域（原版） selected_best_epoch
- PASS: 随机区域（原版） completed_with_test
- PASS: 随机区域（速度优化版） history_matches_checkpoint
- PASS: 随机区域（速度优化版） selected_best_epoch
- PASS: 随机区域（速度优化版） completed_with_test
- PASS: Stage-1 Base dataset_split_matches
- PASS: Stage-2 Base dataset_split_matches
- PASS: Stage-2 Base encoder_source_hash
- PASS: Stage-2 Base frozen_encoder_best_last
- PASS: Stage-2 Base target_scaler_matches
- PASS: 旧多尺度模型 dataset_split_matches
- PASS: 旧多尺度模型 encoder_source_hash
- PASS: 旧多尺度模型 frozen_encoder_best_last
- PASS: 旧多尺度模型 target_scaler_matches
- PASS: 随机区域（原版） dataset_split_matches
- PASS: 随机区域（原版） encoder_source_hash
- PASS: 随机区域（原版） frozen_encoder_best_last
- PASS: 随机区域（原版） target_scaler_matches
- PASS: 随机区域（速度优化版） dataset_split_matches
- PASS: 随机区域（速度优化版） encoder_source_hash
- PASS: 随机区域（速度优化版） frozen_encoder_best_last
- PASS: 随机区域（速度优化版） target_scaler_matches
- PASS: optimization_full_history_exact
- PASS: optimization_test_metrics_exact
- PASS: optimization_model_config_exact
- PASS: optimization_training_protocol_exact
- PASS: optimization_best_weights_exact
- PASS: paired_base_common_training_protocol
- PASS: paired_seeds
