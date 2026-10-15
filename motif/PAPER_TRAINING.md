# 原论文模型＋Motif 模块训练入口

`train_paper.py` 与 `train_property.py` 是不同入口：前者使用实际论文模型类及原输入特征，后者使用独立语境网络。当前先支持与现有 Lipo 数据对应的单目标回归，不冒充论文全部分类、多任务与原始实验协议的复现。

## 输入和划分

- `--data` 使用已准备的 `lipo.pt`；严格沿用其中的 train/valid/test 和 row ID。
- `--checkpoint` 必须与该 data SHA-256 一致，提取方法为对应论文或 both。
- 默认 `--arm pretrained`。`--arm original` 不接入新模块，走同一原模型、数据拼接和训练协议，便于以后比较；本轮只做运行检查，没有开展正式效果比较。
- 训练标签按 train 均值/标准差缩放，验证/测试在原单位计算 RMSE。AdamW、梯度裁剪、验证集 RMSE 选模；辅助项分别使用 HimNet 原图/指纹对齐损失、HMG-MRL 原 atom/motif 对齐损失。这里是明确的新统一回归训练协议，不等同原论文完整统计复现。
- 每个训练分子均参与；最后只有一个分子时并入前一批，以兼容原模型 BatchNorm。测试只在选模后执行。
- `--smoke-per-split N` 仅取每个划分前 N 个真实分子，标记 smoke，不用于性能结论。它不是合成输入。
- 保存原模型与本模块源码哈希、数据哈希、实际行号、超参数、最佳权重和最终预测。源码在运行期间变化会报错。

## HimNet

```powershell
$motifPython = 'C:/Users/12775/miniconda3/envs/polyolefin_ml/python.exe'
& $motifPython -m motif.train_paper --paper HimNet --paper-root coarse_graining1/HimNet_reproduction_20261013/windows --data motif/data/lipo.pt --checkpoint motif/runs/revision_pretrain_smoke/best.pt --output motif/runs/him_real_smoke --smoke-per-split 2 --epochs 1 --batch-size 2 --motif-batch-size 4 --hidden 32
```

正式训练去掉 `--smoke-per-split`、`--hidden 32`，改为正式预训练 checkpoint、新输出目录和所需轮次。

入口复用原 `HiMolGraph` 提取特征、BRICS 成员、指纹和全部索引，再在内存中按原 `BatchMolGraph` 的规则拼接，绕过其硬编码的 cwd 缓存读取。`check_paper_data.py` 使用真实分子对比新旧两种拼接，要求所有模型输入 tensor 逐位相同，不修复原图索引或批间注意力问题。

首次可缓存所选分子的原图到 output/him_graphs.pt；后续用 `--him-graph-cache` 复用。缓存同时核验数据和原代码哈希。`--prepare-only` 只构建、检查输入，不训练。

需要原项目自身依赖（torch_geometric、RDKit、NumPy、pandas、sklearn 等）。上例使用此前可运行的 Windows 兼容版路径；可以指定其他兼容版本，但实际源文件哈希会记录，不将修复差异隐藏。

本轮实际完成上述 6 个真实 Lipo 分子（每划分 2 个）的 HimNet＋模块 1 epoch 运行，包括反向传播、验证集选模、最佳权重重载及最终测试，结果明确标记为 smoke。

## HMG-MRL

```powershell
& $motifPython -m motif.train_paper --paper HMG-MRL --paper-root research_code_audit/HMG-MRL --hmg-feature-cache /path/to/original_features.pickle --data motif/data/lipo.pt --checkpoint motif/runs/pretrain_seed0/best.pt --output motif/runs/hmg_paper_seed0 --epochs 100 --motif-batch-size 16 --pair-budget 262144
```

`--hmg-feature-cache` 是**由原项目预处理得到的可信本地 pickle**，不是新的内部扩散缓存。入口直接执行原 `get_smiles_array` 函数组装该缓存，检查分子键、原子映射、有效 Motif token 和 200 维描述符。任何分子缺少特征都会失败，不静默删除、不用零向量/新二维特征冒充原输入。

官方 README 明确要求外部准备缓存，公开仓库没有附各任务现成缓存。新增 `prepare_hmg.py` 直接调用官方 `save_smiles_dicts`，生成所需缓存，不用替代二维特征填充：

```powershell
# 在自己的兼容环境安装官方预处理依赖；本机使用隔离目录，不改原环境。
& $motifPython -m pip install --no-deps --target motif/vendor/deepchem_runtime deepchem==2.8.0

& $motifPython -m motif.prepare_hmg --data motif/data/lipo.pt --paper-root research_code_audit/HMG-MRL --output motif/runs/hmg_features --dependency-path motif/vendor/deepchem_runtime
```

输出 `features.pickle` 和来源清单、预处理日志，保存 3D 构象、MMFF 参数、DeepChem/RDKit 版本、提取开关及请求的原始行号，并校验所有请求分子都得到合法特征。`--smoke-per-split 2` 只生成 6 个真实分子的检查缓存；正式生成去掉该参数。

这里刻意使用官方 `save_smiles_dicts` 路径：同文件的 `get_smiles_dicts` 在第二轮分子遍历中没有重新绑定 `bond_features`，不应误用。没有修改这两个上游函数。官方构象代码未显式为每次嵌入传 seed，所以记录全局 seed 并不承诺构象逐位重放。

隔离安装的 `--no-deps` 只适用于本机已有 NumPy/SciPy/pandas/sklearn 等依赖的环境；新环境应正常安装原项目依赖。缓存生成需要原项目预处理环境，读取既有缓存不需要 DeepChem。

本轮已使用官方 save_smiles_dicts 实际生成 6 个真实 Lipo 分子的缓存，全部特征和 200 维描述符通过校验，随后跑通 HMG-MRL＋模块的 1 epoch 前向/反向、选模、重载与最终测试。路径：`runs/hmg_original_features_smoke/manifest.json`、`runs/hmg_real_smoke/results.json`。这取代此前“只有合成输入检查”的完成范围；仍不是正式性能训练。

原 HMG 模型包含构象键长和原描述符分支，此入口保留它们。新内部预训练仍仅使用二维信息。若要求整个模型完全二维，需要另行定义、记录原模型的输入修改，当前不擅自这么做。

原 HMG 掩码工具硬编码 CUDA，因此该入口要求 CUDA；没有偷偷改成别的掩码计算。即使提供合法缓存，也需要兼容的原论文运行环境。

## 显存控制

预训练按 Motif 大小分桶；适配器、独立语境网络及导出共用 `encode_motifs`。适配器的输入辅助函数现在返回未 padding 的图列表，再在内部大小分桶、分块、恢复顺序。兼容原来的 dense 字典接口，但调用方若提前构造超大 dense batch，其分配成本无法被编码器追溯消除。

`--pair-budget` 限制每个内部子批 `批大小 × 最大原子数²`；单个 Motif 就超预算时明确报错，要求主动调大预算，不丢弃样本。它是张量规模上限，不是显存字节保证。

有梯度的内部编码使用非重入 activation checkpoint，降低多子批反向前保留的大型激活；重计算增加计算开销。数值检查覆盖分块前后的向量、顺序及梯度一致性。
