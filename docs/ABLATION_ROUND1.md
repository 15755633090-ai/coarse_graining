# 第一轮核心消融：代码与执行范围

当前仅准备代码，不启动或排队新增实验。既有 C/D 补跑沿用原入口。

## 范围

所有版本复用同一个 Frozen Base coarse 母模型、输入投影、prediction head、FP32 冻结特征缓存和原训练器。固定原 Lipo split、hidden dim、AdamW、学习率、batch/micro batch、AMP、200 epochs 上限、patience=30 和 seed 0–4，不增加调参。

| 顺序 | CLI 名称 | 实验定义 |
|---|---|---|
| 已有 C | `region_only` | 原 2 层 Region GNN，无 Coarse GNN；只读取既有结果 |
| 已有 D | `base_coarse` | 原 2 层 Region GNN + 3 层 Coarse GNN；只读取既有结果 |
| 1：A | `atom` | 保留输入投影，直接对原子 mean，再使用共同预测头 |
| 2：B-mean | `direct_mean` | Region/Coarse GNN 均设为 0；core mean → graph mean |
| 检查项 | `direct_weighted` | core mean → size-weighted graph mean；仅供等价检查，不接受训练命令 |
| 3：D-no-edge | `no_edge` | 保留 D 的全部参数和 3 层粗节点自身更新，仅移除粗节点间消息边 |
| 4：Deeper Region-only | `deeper_region` | 将 D 的 3 层 Coarse GNN 移到区域内，形成 5 层 Region GNN；无 Coarse GNN |
| 5：Core-only | `core_only` | 保留 D 的中心、owner、cores、粗边和参数，仅将 Region context 改为各 core 的诱导子图 |

Deeper 直接复用母模型初始化出的 3 个 GIN 层，因此与 D 精确匹配层数及可训练参数量。所有开关均在统一母模型初始化后执行，不额外消耗随机数，公共输入投影和预测头初始权重一致。

Core-only 与已有 D 的 overlap 结果配对；本轮不再额外重跑一份 overlap。若执行全部五个新增版本，就是五种配置 × 五个 seeds；B-weighted 无独立训练。

## B-weighted 检查的含义

当 cores 完整覆盖原子且互斥，两次池化之间没有额外变换时：

\[
\frac{1}{N}\sum_i h_i
= \sum_k\frac{|C_k|}{N}\left(\frac{1}{|C_k|}\sum_{i\in C_k}h_i\right).
\]

代码用合成图验证池化及梯度等价，预检查再在真实训练样本上比较 A 与 B-weighted 的输出。
FP32 使用 `rtol=2e-5, atol=2e-6` 检查；BF16 两级池化会引入额外舍入，因此记录误差，不要求逐位相等。
B-mean 对每个 core 等权，A 对每个原子等权，A→B 的变化包含读出权重变化，不直接等同于信息损失。

## 代码位置

- `scripts/experiments/ablation_models.py`：同一母模型的五个开关与 Core-only 数据视图。
- `scripts/experiments/run_frozen_ablation.py`：原协议/缓存核验、预检查、顺序调用原训练器和五 seed 配对汇总。
- `tests/test_frozen_ablation.py`：合成图检查，不需要实际 Lipo 训练或预训练权重。

根目录原三个训练文件和 `coarse_gnn` 实现保持不变，避免影响正在进行的 C/D 训练和旧断点身份。
Core-only 数据视图不修改原拓扑对象；新预检查只在内存构建拓扑，不写入原实验的拓扑缓存。

## 后续执行命令（本次不执行）

在 `coarse_graining` 根目录运行。

```powershell
# 参数说明
conda run --no-capture-output -n polyolefin_ml python -m scripts.experiments.run_frozen_ablation --help

# 预检查：核对协议/共享缓存，用训练样本检查初始化、梯度和池化等价；不做正式训练
conda run --no-capture-output -n polyolefin_ml python -m scripts.experiments.run_frozen_ablation --action prepare

# C/D 的全部五个 seed 完成后，先只跑 A
conda run --no-capture-output -n polyolefin_ml python -m scripts.experiments.run_frozen_ablation --action train --variants atom --seeds 0 1 2 3 4

# 后续逐项执行时，variants 分别选择 direct_mean、no_edge、deeper_region、core_only
# 只有显式运行下面命令，才会按上述顺序执行全部五个新增版本
conda run --no-capture-output -n polyolefin_ml python -m scripts.experiments.run_frozen_ablation --action train --seeds 0 1 2 3 4

# 汇总目前已完成的结果
conda run --no-capture-output -n polyolefin_ml python -m scripts.experiments.run_frozen_ablation --action summarize
```

训练入口在 C/D seed 0–4 未全部完成时直接拒绝训练，不等待、不排队、不启动其他进程。
默认 action 为 `prepare`；训练必须显式指定 `train`。只在验证集选择最佳 epoch，每次 run 完成后计算 test，汇总不用于自动选配置。
中断后执行相同命令，已完成 run 跳过，未完成 run 沿用原 `resume.pt` 恢复；配置/代码变化会触发身份不匹配检查。
前台入口同一时间只运行一个实例。

## 新结果位置与汇总

新增输出默认写入独立的 `outputs/frozen_ablation_round1/`，不改写原正式结果：

```text
outputs/frozen_ablation_round1/
├── run_config.json                # 原协议、母模型身份、新源码哈希、缓存身份和参数量
├── preflight.json                 # 初始化/梯度/池化等价检查
├── comparison.json               # 各模型五 seed 统计与逐 seed 配对差值
├── per_seed.csv                   # 完整逐 seed 结果及来源
└── lipo/<variant>/seed_<seed>/    # 原训练器的 best/resume/history/result 等文件
```

配对比较包括 A→B、B→C、C→D、D-no-edge→D、Deeper→D、Core-only→D。
差值定义为后者 RMSE − 前者 RMSE，负数表示后者更好；报告逐 seed、均值和样本标准差。
新汇总读取 C/D 的全部五个 seed，不依赖原来只面向 seed 0/1 的 comparison 文件；少于五个结果时明确记录实际数量。

## 本轮停止位置

完成 Core-only 后停止。本入口不包含 random topology、random centers、residual、layers/radius/center-fraction 扫描或 Enhanced，不能自动扩展到第二轮。
