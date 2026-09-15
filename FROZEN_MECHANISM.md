# 第一阶段：冻结编码器的粗图传播机制验证

当前优先运行 **Frozen Region-only / Frozen Base coarse × seed 0、1**，共 4 次训练。暂停先前的 finetune 计划，保留原结果、最近持久化断点以及 `05_coarse_gnn/_recovery/pause_finetune_*` 备份。第一阶段完成后不会自动恢复 finetune，也不运行 Enhanced。

直接读取旧 `04_diffusion/lipo/pretrained_frozen` 结果作为历史对照，不重跑旧模型。主要机制比较是两个新模型之间的差异。

## 固定设置

- 完整复用原 Lipo 数据、清洗、样本顺序和官方 scaffold split（3360 / 420 / 420）。
- 相同原预训练 encoder，所有 encoder 参数 `requires_grad=False`，始终 `eval()`，训练时无 encoder 优化器参数组。
- 采用旧 frozen baseline 在验证集选定的同一组参数：head LR=0.001、dropout=0.1、有效 batch=32、micro batch=8。两种新模型不分别调参，新增搜索次数为 0。结果中的旧格式 `encoder_lr` 字段保留占位值；encoder 的实际学习率为 0，因为它完全不进入优化器。
- 相同 train-only target scaler、AdamW、weight decay、梯度裁剪、200 epochs 上限、patience=30、原下游 BF16 AMP 与随机种子控制。
- radius=4、center_fraction=0.1、max_residual_size=4、canonicalize=True。
- 两者均关闭区域边特征、粗边计数/特征、区域大小特征，使用 mean graph pooling。Region-only 的 coarse_layers=0，Base coarse=3；其余共同模块在同 seed 下初始化完全一致。
- 按 seed 0 的两个模型、seed 1 的两个模型执行，便于较早得到配对结果。只由 validation 选择 best checkpoint；test 在各 run 完成后按原训练器计算，不参与参数选择。

## 固定表示缓存

`frozen_features.py` 用 eval-mode encoder、关闭 autocast 的 FP32 前向一次性提取每个分子的原始节点表示，写入 `_encoder_cache`。两个模型、两个 seed、训练和验证均读取同一套张量。缓存绑定原 checkpoint 哈希、样本拓扑指纹和顺序、split 身份、结构配置、提取批大小、设备及 PyTorch 版本；更换输入或 checkpoint 会使用新缓存文件。

缓存提取不拟合标签，也不使用性质预测头。冻结机制训练不再运行扩散编码器前向或反向，只有新增模块参与训练。拓扑缓存继续复用 `05_coarse_gnn/_topology_cache`。

历史 frozen baseline 在训练时沿用了整模型 autocast；这里将 encoder 特征统一固定为 FP32，存在执行精度差别。因此严格的“粗图传播贡献”结论来自两组新模型的配对比较，不能把与旧 baseline 的差异全部归因为粗化。两组新模型之间没有这一差别。

## 启动与恢复

在 `coarse_graining` 根目录运行。默认只准备，训练需显式 `--action train`：

```powershell
conda run --no-capture-output -n polyolefin_ml python -u run_lipo_frozen.py --action prepare --seeds 0 1
conda run --no-capture-output -n polyolefin_ml python -u run_lipo_frozen.py --action train --seeds 0 1
```

本机也可使用隐藏窗口后台脚本（与前台命令二选一）：

```powershell
.\start_lipo_frozen.ps1
```

原 `start_lipo_optimized.ps1` / `run_lipo_formal.py` 属于暂停的 finetune 实验，现在不要同时运行。中断后重新运行 frozen 命令，已完成结果跳过、未完成 run 从原 `resume.pt` 接续。代码或协议改变会拒绝混入同一结果目录。

## 文件与解释

```text
model/results_formal/05_coarse_gnn/frozen_mechanism/
├── _logs/
├── _encoder_cache/
├── run_config.json
├── preflight.json
├── downstream_results.json
├── comparison.json
├── comparison_seed_0_1.csv
└── lipo/
    ├── region_only/seed_0, seed_1/
    └── base_coarse/seed_0, seed_1/
```

每个 run 沿用 `best.pt`、`history.json`、`run_config.json`、`result.json`、`test_predictions.csv`，训练中另有 `resume.pt` 和每轮更新的 `progress.json`。根目录配置记录旧参数选择文件哈希、新增搜索为 0、encoder 冻结、精度策略和实际源码身份。

`comparison.json` 的 `rmse_base_minus_region` 为负表示该 seed 下粗图传播更好；先检查两个 seed 的方向，不能把 2 seeds 当作最终论文统计。即使这一阶段支持后续实验，也需要另行启动第二阶段 finetune 性能验证，不自动推进。
