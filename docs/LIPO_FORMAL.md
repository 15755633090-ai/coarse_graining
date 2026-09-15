# Lipo 粗粒度模型正式协议接入

> 本页命令均在 `coarse_graining` 根目录执行，工具脚本的新位置见 [目录与命令索引](DIRECTORY_LAYOUT.md)。

**当前计划已调整为先做 [Frozen 机制验证](FROZEN_MECHANISM.md)。本页记录的 finetune 实验已暂停，已有结果和断点保留；以下 finetune 启动命令现在不执行。**

入口为 `run_lipo_formal.py`。默认 `--action prepare` 只做一致性检查和拓扑预计算；显式 `--action train` 才启动调参及正式训练。结果写入项目旁的 `model/results_formal/05_coarse_gnn`。已有 `01_local`、`02_dmpnn`、`03_grover`、`04_diffusion` 只读。

## 复用范围

直接导入 `扩散GNN/01_扩散预训练/downstream_benchmark.py`，通过临时替换模型工厂调用原 `main()` 和 `run_single()`。原数据加载、SMILES 处理、官方 split、DataLoader、train-only scaler、损失、优化器、AMP、随机数控制、早停、断点恢复、验证选模、测试隔离、预测 CSV 和汇总函数均直接执行原代码；没有复制一套新训练器，也没有新划分。

读取 `04_diffusion/lipo/pretrained_finetune/selected_config.json` 中的完整历史协议和 `04_diffusion/run_config.json`。启动时核对旧训练源码、数据文件、split 和预训练 checkpoint 的哈希；不使用 coarse demo 中的导出编码器替代原正式 checkpoint。

固定设置：

| 项目 | 本轮设置 |
|---|---|
| 数据 | Lipo，原 OGB scaffold split，3360 / 420 / 420 |
| 调参 | 独立 seed 42，每个新模型 4 个原候选 |
| 正式训练 | 当前 seed 0、1；以后可以补 2、3、4 |
| 最大 epoch / patience | 200 / 30 |
| micro batch | 8；有效 batch 由原候选决定 |
| 优化器 | 原 AdamW，weight decay=1e-5，grad clip=5 |
| 执行 | CUDA、原 BF16 AMP、严格确定性算法 |
| checkpoint 间隔 | 100 batches：历史 run_config 未记录此项，沿用原源码默认值，不使用示例中的 50 |
| 粗化 | radius=4，center_fraction=0.1，max_residual_size=4，canonicalize=True |
| 区域层 / 粗图层 | 2 / 3；仅 Region-only 的粗图层为 0 |
| Encoder | 同一原预训练 checkpoint，全部 finetune，解码头和旧 graph projection 不参与 |

保持原 4 个候选的 batch size、encoder LR、head LR 和 dropout，不增加结构搜索。三种新模型各在验证集独立选择最优配置，然后锁定该配置训练正式 seeds。所有新增区域/粗图/预测参数进入原优化器的 head 参数组。

## 模型与比较

- `region_only`：使用 Base 配置，关闭所有增强项，mean readout，`coarse_layers=0`。
- `base_coarse`：与 Region-only 的结构配置仅相差 3 层粗图传播。相同 seed/候选下公共 encoder、region 和预测头初始化一致。
- `enhanced_coarse`：保留 region bond features、coarse count/bond features、size feature 和 size-weighted readout。

旧 Diffusion `pretrained_finetune` 是最直接的编码器对照；Morgan RF/XGBoost、GINE、AttentiveFP、PNA、D-MPNN、GROVER 和其余 Diffusion 状态仍参与整体性能比较，不重新训练。

重点看 Base coarse − Region-only 的逐 seed test RMSE 差值，负值表示 Base 更好。由于沿用各模型独立调参协议，如果两者选中的超参数不同，`comparison.json` 会记录 `matched_hyperparameters=false`；此时差异不能严格全部归因于粗图传播。2 个 seeds 仅用于探索趋势。

## 启动

在 `coarse_graining` 目录运行：

```powershell
conda run --no-capture-output -n polyolefin_ml python run_lipo_formal.py --action prepare --seeds 0 1
conda run --no-capture-output -n polyolefin_ml python run_lipo_formal.py --action train --seeds 0 1 2>&1 | Tee-Object -FilePath "..\model\results_formal\05_coarse_gnn\_logs\lipo_seed_0_1.log" -Append
```

`train` 启动前仍自动做一致性检查，防止准备后数据或模型文件被替换。本轮执行 **3 模型 × 4 调参候选 = 12 次调参 + 3 模型 × 2 seeds = 6 次正式训练**。在单张 8 GB GPU 上顺序执行。没有为旧基准增加任何训练任务。

中断后重新运行同一命令：复用已完成调参/正式结果，未完成的 run 使用原 `resume.pt` 恢复。后续补 seeds：

```powershell
conda run --no-capture-output -n polyolefin_ml python run_lipo_formal.py --action train --seeds 2 3 4
```

根配置中的 `seeds` 表示允许的正式 seed 池 0–4；真正执行的 seeds 由当前命令及各 run 的结果记录决定。补 seeds 不重新调参。结构和原协议变更仍触发配置不匹配检查。本次执行优化通过下述明确的版本迁移记录接续旧断点。

## 结果

```text
05_coarse_gnn/
├── _logs/
├── _topology_cache/                  # 全数据集结构缓存；不保存激活或标签
├── preflight.json                   # 数据/顺序/scaler/checkpoint/GPU 检查
├── run_config.json                  # 首次启动 train 时生成
├── downstream_results.json          # 新模型各 seed 结果及汇总
├── downstream_summary.csv
├── comparison.json                  # 旧基准 + 新模型 + 配对 RMSE 差值
├── comparison_seed_0_1.csv           # 各模型共同 seed 0、1 的汇总
└── lipo/
    ├── region_only/
    ├── base_coarse/
    └── enhanced_coarse/
        ├── tuning/seed_42/trial_0..3/
        ├── selected_config.json
        ├── seed_0/
        └── seed_1/
```

每个正式 run 沿用原格式：`run_config.json`、`best.pt`、`history.json`、`result.json`、`test_predictions.csv`。调参 run 的 `test_metrics=null`，不生成 test predictions。比较汇总只读取旧基线 JSON，不移动或覆盖其权重及结果。

一致性检查使用原训练集的一小批数据做重复初始化、AMP 前向/反向和一步优化器诊断；这不构成正式训练，不生成任何新 split，也不计算诊断测试指标。单元接入测试另用合成图验证原训练器的验证选模与测试隔离。

当前整体 32 项测试通过，真实 Lipo 一致性检查通过。4200 个样本全部保留并预计算；相同输入图可以共用缓存条目，因此缓存文件数可以少于样本数。正式训练需显式运行上述 `--action train` 命令。

## 执行优化与断点接续

`coarse_gnn/packed.py` 把一个 micro batch 中的区域和粗图组成互不连通的批图，集中执行原有 GNN 参数、主归属 pooling 和图读出。不同分子之间没有新增边，编码器仍在每次训练中计算并反向传播，不缓存会随微调变化的节点表示。

`coarse_gnn/prepared_data.py` 在内存中复用原数据集生成的分子特征与已校验的 CPU 拓扑；collate 和 micro-batch slicing 给原批次附加合批索引。后续 epoch 不再反复解析 SMILES、检查 GPU 上的图结构或执行每张图的 CPU/GPU 索引搬运。第一次加载及恢复时跳过的批次仍有一次准备开销。DataLoader 的样本顺序、有效 batch、micro batch、损失、优化器、学习率、AMP、早停和调参预算均保持原设置。

dropout 按原图/区域/层顺序抽取 mask，随机数状态对照一致。合批改变矩阵梯度的浮点累加顺序，所以 **BF16 训练轨迹不保证与旧串行实现逐位一致**。真实权重/输入检查中 FP32 最大预测差约 2.4e-7；BF16 预测在所测批次中一致，整体梯度相对 L2 差约 0.36%–0.57%。不能据此推断完整训练后的预测也逐位相同。

本机 8 分子微批前向＋反向实测约快 6.3–6.7 倍。这不等于整轮或完整实验的加速倍数；原编码器、数据准备、验证和 checkpoint I/O 仍有开销。测量结果见 `outputs/performance/packed_audit.json`。`scripts/performance/benchmark_resumed_epoch.py` 另在临时副本中恢复真实第 26 轮第 100 批并完成该轮；该次只剩 5 个训练批次，包含首次分子缓存准备，不能用于估算完整热缓存 epoch 的速度。副本恢复后的 validation RMSE 差约 1.5e-5，原断点文件哈希未变，未访问 test。

`coarse_gnn/execution_revision.py` 只允许已审计的串行源代码快照切换到通过当前哈希核验的合批版本；任何超参数、split 或其他协议字段变化都会拒绝迁移。旧 `run_config.json` 与调参协议身份保留，根 `execution_revision.json` 和每个 run 的 `execution_history.json` 单独记录实际执行源码、审计哈希、恢复断点哈希及时间。模型权重和优化器状态布局未改变。

优化代码发生变化后，必须先重新运行数值审计；未通过或源码哈希过期将拒绝训练：

```powershell
conda run --no-capture-output -n polyolefin_ml python -m scripts.validation.verify_packed
```

优化前源码保存在 `outputs/performance/reference`，便于复核本次接续的来源。以上检查不能替代正式多 seed 性能实验。

本机还提供后台接续脚本 `scripts/launchers/start_lipo_optimized.ps1`：检查是否已有 Lipo 训练进程，备份原 Region-only trial_0 断点，在隐藏窗口接续 seed 0、1，并返回 `_logs/lipo_optimized_时间戳.log` 路径。前台命令和后台脚本选一种使用，避免重复启动。关闭查看日志的窗口不影响后台训练。

```powershell
.\scripts\launchers\start_lipo_optimized.ps1
```
