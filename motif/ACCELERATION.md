# 预训练加速与恢复

本次只优化计算和保存方式，不改网络、逐 Motif 损失定义、噪声日程、噪声分组、batch 划分、训练集或验证选模准则；不使用混合精度。

## 修改

- 损失：从逐 Motif 的上百次小交叉熵及 GPU 布尔同步，改为每属性一次批量交叉熵、掩码和逐 Motif 分层规约。返回的 atom/edge 统计为 detached tensor，需要打印时再取标量。
- 验证噪声：仍使用同一个按样本身份固定的 GPU RNG 生成；一次生成后缓存至有限的 pinned CPU 内存，每轮复用。`--validation-cache-mb 512` 是缓存上限，超预算批次仍走原生成路径，不丢弃样本、不改用 CPU 随机数。
- 恢复指标：在设备上批量归约整数计数，只传回一个小计数表，保留完全相同的分母、t=0 去重和方法计数。
- 日志：补充训练/验证耗时及每 100 个训练 batch 的 progress.json，减少逐 batch 标量拷回 CPU。
- 每轮原子保存 `last.pt`：当前模型、AdamW 状态、CPU/CUDA RNG、历史、最佳权重哈希及协议。`best.pt` 继续供表示提取与下游使用。

## 续训

```powershell
python -m motif.pretrain --data motif/data/lipo.pt --output motif/runs/new_resume_dir --resume-from motif/runs/original_dir/last.pt --method both --seed 0 --hidden 128 --layers 3 --steps 100 --epochs 50 --batch-size 32 --pair-budget 262144
```

恢复会校验数据、方法、结构、种子、lr、batch/预算等，并同时恢复优化器和 RNG。允许更换 CPU 验证缓存上限，因为它不改变噪声；不得把 weight-only 的 `best.pt` 当作完整续训检查点。小规模测试已验证新实现连续 2 epoch 与 1 epoch 保存后恢复，CPU 模型、RNG 和损失逐位一致。CUDA 浮点规约、不同设备和不同实现之间不承诺轨迹逐位一致。

## 旧运行处理

旧 `pretrain_lipo_both_seed0_20261015` 保存的是权重型 best.pt，没有优化器/RNG。它完成了 9 个 epoch，全部日志与最佳权重保留；用户要求加速后停止并写入 interrupted_for_acceleration.json。为了避免默默重置 Adam 后声称无缝续训，加速版采用原 seed 0、原参数从头重跑 50 epoch，新目录独立记录，不能把两段历史拼接为同一次运行。

## 检查与测速边界

`check_acceleration.py --reference-root <旧冻结源码目录> --output <新的检查目录>` 验证旧/新损失与梯度、恢复整数计数、缓存噪声和完整检查点恢复。参考源码哈希写入结果。

本机在停止旧任务后，128 维、3 层、batch=32 的代表性 GPU 训练步测试：

| 每个图的原子数 | 旧版秒/步 | 加速版秒/步 | 单步速度比 |
|---|---:|---:|---:|
| 8 | 0.13926 | 0.01295 | 10.76× |
| 24 | 0.20607 | 0.01913 | 10.77× |

这些是独立的合成训练步测速，不是整轮加速倍数或性能收益。实际全量 epoch 的训练、验证耗时会写入 history.json；需据此报告整轮速度。所有检查和测速均不评价预训练恢复质量或分子性质预测增益。

加速版真实全量运行的首个 epoch 已完成：训练 170.83 秒，验证 29.39 秒，总计 200.22 秒，约 3 分 20 秒。旧版此前约 7 分 30 秒/轮，因此初步整体速度约为原来的 2.2 倍；这是首轮与此前平均耗时的粗略比较，启动阶段有其他实现检查并行，不能视为严格配对测速或稳定 ETA。后续以实际 history.json 为准，不把单步 10.8× 宣传成整轮速度。
