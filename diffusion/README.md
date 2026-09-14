# Diffusion：扩散预训练与编码器模块

新方向的起点：分子图离散扩散预训练与编码器。当前尚未实现新的粗粒度映射或训练目标。

本目录是 `coarse_graining` 方法项目的扩散模块，由 `../../扩散GNN/01_扩散预训练` 提取，可独立加载权重、编码分子和运行扩散预训练。原目录保持不变。以下命令均在本模块目录执行。

## 保留内容

- `bond_diffusion/`：原子/键数据表示、加噪、去噪模型、化学损失、训练工具及预训练评估。
- `train.py`：扩散预训练入口。
- `outputs/ogb_clean/best.pt`：正式预训练最佳权重，第 29 轮；完整训练共 30 轮。
- `outputs/ogb_clean/encoder.pt`：从上述最佳权重导出的推理文件，不含优化器；为兼容原加载接口，仍包含去噪头参数。
- `outputs/ogb_clean/last.pt`、`resume.pt`、`history.json`：原训练检查点和历史。
- `outputs/ogb_clean/formal_recovery_v2/`：正式恢复评估结果。`formal_recovery/` 是保留的较早评估版本。
- `outputs/ogb_clean/structural_probe_v2/`：冻结编码器的结构探测结果。
- `datasets/`：423,114 条 clean 预训练分子的 CSV、恢复/结构探测数据及来源清单。

不包含节点回溯、根节点选择、性质软子图、区域选择器、C2/C3 或下游性质预测实验。模型中后加的外部消息门控接口和逐层回溯消费者接口已移除；原始消息传递层内部的可学习 gate 属于扩散编码器结构，保留以兼容权重。

约 2 GB 的预处理 JSONL 缓存没有重复复制，训练可直接读取本目录 CSV；如需严格重放旧的中断训练，应使用原训练使用的同一数据文件及配置。历史结果 JSON/清单中的旧绝对路径作为来源记录保留，不是当前运行入口。

## 环境

使用已有环境：

```powershell
conda activate polyolefin_ml
cd 'C:\Users\12775\Desktop\GNN课题1\coarse_graining\diffusion'
```

## 使用预训练编码器

正式模型为 4 层消息传递，节点隐藏维度 128。默认在扩散时间步 0 编码干净分子图。

```python
import torch
from bond_diffusion.data import collate_graphs, graph_from_smiles
from bond_diffusion.trainer import load_encoder

encoder = load_encoder('outputs/ogb_clean/encoder.pt').requires_grad_(False)
batch = collate_graphs([graph_from_smiles('CC(=O)O')])
with torch.no_grad():
    h = encoder.encode_nodes(batch.node_features, batch.bonds, batch.node_mask)
    # h: [batch_size, padded_num_atoms, 128]
    # batch.node_mask 标识有效原子。
```

批量命令行示例（输出原子表示及 sum/mean 拼接的 256 维图表示）：

```powershell
python encode_smiles.py 'CCO' 'CC(=O)O' --output outputs/example_embeddings.pt
```

原来的 `model.encode()` 及图投影参数也保留，便于兼容旧结果。但原预训练损失没有使用 `graph_embedding`，所以该投影层没有通过此损失训练；新方向建议从 `encode_nodes()` 的节点表示开始，按需要设计粗粒度聚合。

## 扩散预训练

轻量运行检查：

```powershell
python train.py --demo --epochs 1 --hidden-dim 32 --num-layers 1 --device cpu --output-dir outputs/demo
```

从头预训练，架构与已有正式权重一致，输出写入新目录：

```powershell
python train.py --input datasets/ogb_pretrain_clean.csv --epochs 30 --hidden-dim 128 --num-layers 4 --device cuda --output-dir outputs/new_pretrain
```

导出新训练的推理权重：

```powershell
python export_encoder.py outputs/new_pretrain/best.pt outputs/new_pretrain/encoder.pt
```

`evaluate_formal_recovery.py` 和 `structural_probe.py` 保留用于评估扩散恢复与结构表示，参数可通过 `--help` 查看。原模型是使用 GNN 去噪器的原子/键离散扩散预训练模型。

## 复制验证

`copy_manifest.json` 记录复制来源、文件 SHA256 和修改范围。`verification.json` 记录实际验证结果：复制文件校验、原模型与精简模型输出一致、导出后重新加载一致，以及扩散损失反向传播通过。

`verify_copy.py` 是一次性的来源对照检查工具；只有它需要原目录，训练和编码入口均不依赖原目录。

编码器核验详见 `ENCODER_AUDIT.md`。已在本副本修复“全为单原子分子的批次产生空键交叉熵 NaN”的边界情况。已有权重没有修改。

```powershell
python -m unittest discover -s tests -v
```
