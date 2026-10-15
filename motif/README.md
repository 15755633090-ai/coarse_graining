# 二维 Motif 扩散预训练与分子语境学习

独立克隆仓库时，先阅读 [GITHUB_SETUP.md](GITHUB_SETUP.md)；原论文真实数据训练见 [PAPER_TRAINING.md](PAPER_TRAINING.md)。

当前为第一版可运行实现，不代表已经建立预测增益。全部新代码、缓存及小规模运行产物放在本目录；没有修改原论文仓库、既有 Encoder、划分及实验结果。

## 已实现的路径

```text
已有审计过的 Motif 成员索引
    → 内部诱导图：原子类别 + 所有无序原子对的键类别（包括无键）
    → 固定 one-hot 编码 + 多时间步高斯噪声
    → 时间条件的节点—边联合更新
    → 原子属性 / 有无键 / 键类型分类恢复
    → t=0 的节点与边联合读出 s_M
    → 保留所有原子与化学键，增加 Motif 节点及双向成员边
    → 分子语境消息传递 → 分子性质预测
```

并提供 HMG-MRL、HimNet 的 Motif 初始表示替换适配器。**独立语境模型**与**论文适配器**是两种使用方式，不把前者的结果称为完整论文复现。HMG-MRL 适配器保留其原有 Cross-Attention/Transformer 路径；HimNet 适配器保留其原有扩展图与融合路径。只有独立语境模型采用本目录实现的化学键/成员边关系算子。

## 文件

| 文件 | 功能 |
|---|---|
| `prepare.py` | 复用 verified_motifs.json，核对 SHA-256，保留两套原始成员与 train/valid/test 标记 |
| `data.py` | 诱导图特征、padding、有效无序原子对掩码 |
| `diffusion.py` | 噪声日程、节点—边联合去噪、分类损失、干净图编码 |
| `pretrain.py` | 单个共享模型预训练，验证集固定噪声选模，保存可重载权重 |
| `batching.py` | 共用大小分桶、原子对预算、分块编码与逐样本固定验证噪声 |
| `context.py` | 保留原子及所有化学键，处理重叠 Motif 和未覆盖原子 |
| `train_property.py` | Lipo 回归入口，验证集选模后只做一次最终测试 |
| `train_paper.py`、`paper_data.py` | 原论文模型类的真实分子训练入口与数据桥接；见 PAPER_TRAINING.md |
| `prepare_hmg.py` | 调用官方 save_smiles_dicts 生成原模型特征缓存，校验缺失分子及特征维度 |
| `adapters.py` | 两套论文模型的表示替换与原始节点/token 顺序绑定辅助函数 |
| `export.py` | 导出 t=0 Motif 向量及 row/method/group/member 映射 |
| `checks.py` | 数值、掩码、梯度、排列一致性和适配接口检查 |
| `check_revisions.py` | 逐基序损失与验证噪声的分批不变性、预算及分块梯度检查 |
| `recovery.py`、`check_recovery.py` | 按噪声分层的恢复指标、完全恢复率及计数检查 |
| `check_paper_data.py` | 真实分子的 HimNet 输入与原版拼接逐位一致性检查 |
| `check_paper_adapters.py` | 实际论文模型类的合成输入前向/反向接入检查 |

## 固定的实现选择

- 不用三维构象，不用已有扩散 Encoder；本模块从头预训练。
- 内部节点特征：元素、形式电荷、芳香性标记、诱导子图内度数。省略全分子度数、氢数、杂化、环标记和手性，避免直接复制这些外部环境相关属性。芳香性和形式电荷仍是原分子的化学注释，因此不宣称输入完全独立于原分子化学环境；计算过程中不读取外部节点或键。
- 键类别：无键、单、双、三、芳香、其他。第一版没有额外重建立体键类别。
- 固定类别 one-hot 后加高斯噪声，余弦信号日程；默认 100 步，10% 输入为真正干净的 t=0。
- 对每对原子只采一个边噪声，复制到反方向；不加自环，padding 不参与传播或损失。
- 去噪网络不接收干净邻接表；所有有效原子对都进入网络。每个 Motif 单独计算原子属性平均损失，并在该 Motif 内对有键、无键分别求均值再等权组合；最后对 Motif 平均。单原子 Motif 的边项为零，缺失分层不补造样本。训练与验证采用同一口径。
- 验证噪声按 row/method/index 固定 RNG，改变分批方式不改变该样本噪声；验证总分按实际 Motif 数累加，不平均不同大小 batch 的均值。
- 节点、边池化均进入 Motif 表示；解码头使用该表示，保证读出参数实际受到预训练监督。仍不保证读出无损保留拓扑。
- 下游以 t=0 一次前向提取表示；没有随机噪声生成或反向采样过程。
- 重叠 Motif 原样保留；成员边按原子覆盖次数归一化。没有为了补齐覆盖而生成假 Motif，未覆盖原子仍走完整原子图。
- 预训练默认只用原划分 train 分子的 Motif；valid 用于重建选模，test 不参与预训练。相同化学片段可能跨分子/划分重复，当前不做结构去重，不把重建验证当作对全新 Motif 的泛化评估。
- 允许两种方法实例联合预训练一个模型，也可用 `--method` 分开训练；不为每个 Motif 单独训练模型。

## 运行环境

本机已使用 Python 3.9、PyTorch 2.8.0+cu128、RDKit 2025.03.5 运行。独立路径不需要导入原论文的项目依赖，论文适配器则需要原模型自身环境。

在工作区根目录运行 PowerShell：

```powershell
$motifPython = 'C:/Users/12775/miniconda3/envs/polyolefin_ml/python.exe'
& $motifPython -m motif.checks
& $motifPython -m motif.prepare --output motif/data/lipo.pt
```

已有 `data/lipo.pt`：4,200 个分子，HimNet 23,900 个、HMG-MRL 60,866 个原始 Motif，总计 84,766。数据源及特征定义见 `data/lipo.json`。复用了原子成员提取结果，未重新修改化学规则。

### 正式训练入口（尚未正式运行）

```powershell
& $motifPython -m motif.pretrain --data motif/data/lipo.pt --output motif/runs/pretrain_seed0 --epochs 50 --batch-size 32 --hidden 128 --layers 3 --steps 100 --seed 0

& $motifPython -m motif.train_property --data motif/data/lipo.pt --checkpoint motif/runs/pretrain_seed0/best.pt --output motif/runs/property_hmg_seed0 --method HMG-MRL --arm pretrained --seed 0

& $motifPython -m motif.train_property --data motif/data/lipo.pt --checkpoint motif/runs/pretrain_seed0/best.pt --output motif/runs/property_him_seed0 --method HimNet --arm pretrained --seed 0
```

以上性质入口训练的是独立原子+Motif 语境网络。默认联合微调；`--freeze-pretrained` 冻结内部去噪编码器。`--arm random` 使用相同配置的随机内部编码器，`--arm atom-only` 不使用 Motif；提供这些入口不代表本轮已开展新一轮消融。

输出目录中已有 best.pt 时会报错，避免覆盖结果。默认不截断 Motif 原子数；稠密原子对复杂度是 O(N²)，显存不足时先减小 batch-size。下游内部编码按 Motif 大小排序、每 32 个一组，再恢复原顺序，可用 `--motif-batch-size` 调整，避免整批重叠 Motif 都 padding 到最大骨架。没有自动删除大 Motif 的行为。

修订版已将同一大小分桶逻辑用于预训练、两个论文适配器和导出；适配器不再由辅助函数预先拼接整批 dense 图。有梯度时使用 activation checkpoint 减少保留的大型激活。预训练与论文入口可设置 `--pair-budget` 限制每个子批 padded 原子对数，单个 Motif 超预算会报错，不丢弃。

原论文真实数据入口见 [PAPER_TRAINING.md](PAPER_TRAINING.md)。不能用上面的独立 `train_property` 命令结果声称原论文加模块的收益。

正式预训练同时输出 `recovery_metrics.json`：t=0 单独记录，低/中/高噪声分别报告原子属性、键召回、无键误报、键类型及完整 Motif 恢复率。指标定义、分母和解释边界见 [RECOVERY_METRICS.md](RECOVERY_METRICS.md)。它们不改变选模，不等同化学合法性、s_M 信息完整性或下游增益。

### 小规模流程检查

```powershell
& $motifPython -m motif.pretrain --data motif/data/lipo.pt --output motif/runs/smoke_pretrain --epochs 1 --max-batches 2 --batch-size 4 --hidden 32 --layers 2 --steps 20
& $motifPython -m motif.train_property --data motif/data/lipo.pt --checkpoint motif/runs/smoke_pretrain/best.pt --output motif/runs/smoke_property --method HMG-MRL --epochs 1 --max-batches 2 --batch-size 8 --hidden 32 --context-layers 2
```

这两个命令已经跑通，运行记录明确为 `smoke: true`，不能作为性能结果。`--max-batches` 在最终版本限制训练、验证及测试的批数，仅用于实现检查；正式运行不设置该参数。运行目录已存在时，重复检查须更换输出目录。

### 特征导出

```powershell
& $motifPython -m motif.export --data motif/data/lipo.pt --checkpoint motif/runs/pretrain_seed0/best.pt --output motif/runs/pretrained_vectors.pt
```

导出包含精确原子成员和顺序。不能直接假设两套论文的 token/node 顺序与此缓存顺序相同。

## 接入原论文代码

加载去噪器：

```python
from motif.pretrain import load_encoder
denoiser, metadata = load_encoder(checkpoint_path, device)
```

### HMG-MRL

```python
from motif.adapters import HMGMRLAdapter, hmg_motif_inputs
adapted = HMGMRLAdapter(original_backbone, denoiser).to(device)
motif_batch, positions = hmg_motif_inputs(rdkit_mols, original_fragment_maps, device)
outputs = adapted(motif_batch, positions, **original_forward_inputs)
```

`original_fragment_maps` 是原预处理中 `frags_atom_bond_indices_dict`，成员必须仍对应原 RDKit 编号，不能误用按度数重新排序后的节点编号。辅助函数按原论文预处理的 `whole_mol`、`frag_1`… 顺序绑定。只替换真实 Motif token 的初始化，保留 whole_mol 和 padding；后续原子/键 GNN、Motif Transformer、对齐损失和融合由原 backbone 执行。

适配器内部包装 `frag_input_projection`，因此必须传入独立实例，原 checkpoint 应在包装前加载。之后训练保存 adapted.state_dict；重载时先重建相同包装，再 strict=True 加载。ContextVar 在前向异常后也会清理；不要把包装后的 backbone 与未包装基准共享。

### HimNet

```python
from motif.adapters import HimNetAdapter, himnet_motif_inputs
adapted = HimNetAdapter(original_backbone, denoiser).to(device)
motif_batch, positions = himnet_motif_inputs(rdkit_mols, original_cliques, original_batch_graph, device)
pred = adapted(original_batch_graph, motif_batch, positions, training=True)
aux_loss = adapted.compute_contrastive_loss()
```

`original_cliques` 必须按实际 BatchMolGraph 的 BRICS Motif 顺序提供。适配器复制图容器后替换 f_atoms 中的真实 Motif 行，不改原图、原子、全局节点、化学键、成员关系索引或指纹；再执行原 encoder、指纹融合与预测头。调用方负责原模型监督损失与辅助损失加权。支持整批无 Motif。

**接口检查及实际模型类的合成输入前向/反向已完成；正式性能训练尚未执行。** 新增的真实数据入口及当前核验范围见 PAPER_TRAINING.md。HMG 合成检查直接使用 research_code_audit 中模型类；HimNet 使用此前可运行的 Windows 兼容版类。两者输出均为 2×1，梯度均返回内部 Motif 编码器。单独进程执行命令如下，以避免原项目同名顶层模块冲突：

```powershell
& $motifPython -m motif.check_paper_adapters --paper hmg
& $motifPython -m motif.check_paper_adapters --paper him
```

结果保存到 `runs/hmg_adapter_check.json`、`runs/him_adapter_check.json`，包括实际源码路径和哈希。HMG 原版掩码工具硬编码 CUDA，此检查需要 GPU，未在适配器里暗中改写。

当前仓库已审计出的索引/批注意力行为不会被本适配器暗中修复。若要训练完整论文版，应使用此前可运行的本地兼容环境，并明确其兼容差异。完整模型本身可能使用其他特征，二维限制在这里特指新内部预训练模块，不代表原 HMG-MRL 的所有分支都已去除构象依赖。

## 完成状态

- 输入缓存、联合去噪器、干净表示提取、独立语境网络及回归训练入口：已实现。
- 两套论文初始化替换适配器及顺序绑定辅助函数：已实现。
- 实现检查、两套实际模型类适配，以及极小规模 GPU 前向/反向/保存/重载：已通过。
- 修订后的逐 Motif 损失、验证噪声分批不变性，以及分桶/分块的输出和梯度一致性：已通过。
- 原论文回归训练入口：已补充；HimNet 复用真实图构建，HMG-MRL 调用官方脚本生成并读取原版特征缓存。两者均已在 6 个真实 Lipo 分子上跑通 1 epoch 运行检查；不只是合成输入适配。实际核验状态见 PAPER_TRAINING.md。
- 大规模无标签预训练、正式效果比较、预测增益结论：尚未完成；本轮不声称有效。
