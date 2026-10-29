# BRICS 结构引导、边界交互感知的原子级条件去噪

本模块是 2026-10-29 新框架的第一版可运行实现。借鉴 MotiL diffusion priming 的思路，自行实现受损输入、时间条件、逐原子噪声预测及一致的恢复公式。**不是直接沿用 MotiL 官方 DDPM/CMPNN，也不是对其全部方法的复现。** 与 `motif/context_denoising` 的旧基序隐向量模型完全分开，旧 ESOL 结果不属于本框架。

## 已实现的计算路径

```text
完整原子图 + 干净键/拓扑 + BRICS 内部边/跨边界边/边界标记
    ↓
受损原子属性 x_t + 逐原子对齐的时间嵌入
    ├─ s：只沿片段内部边传播
    └─ h：内部传播 + 逐条真实跨边界消息 × 条件软门控
    ↓
逐原子融合 z_v
    ├─ 预训练：逐原子 epsilon 预测 → 单一扩散 MSE
    └─ 微调：t=0 干净输入 → 原子均值读出 → 性质预测
```

不创建基序节点、不池化成基序向量、不把片段编号输入嵌入。编号只用于建图和合批；编码器实际只使用内部/跨边界关系及边界标记。两种状态在同一个编码器、同一次预训练中联合更新。

| 文件 | 功能 |
|---|---|
| data.py | RDKit 图与非重叠 BRICS 连通分量；真实双向连接端点 |
| diffusion.py | 累计 alpha_bar 日程、加噪、epsilon 到 x0 的一致转换 |
| model.py | 双状态原子 MPNN、时间嵌入、边界门控、节点解码与性质读出 |
| training.py | 共享优化/评价、原子分组损失、带重试的原子文件替换、检查点与恢复 |
| pretrain.py | 无标签语料的单一条件扩散预训练 |
| finetune.py | 编码器严格迁移、全参数微调或从头训练对照 |
| test_interfaces.py | 12 项科学接口不变量测试 |

## 第一版固定选择与信息边界

- 原子目标为 133 维固定 one-hot：元素 119 类（含其他）、形式电荷 12 类（-5..5 及其他）、芳香性 2 类。没有输入全分子度数、总氢数、全分子价态；第一版未编码原子手性、同位素，不能称完整化学表征。
- 干净键特征共 16 维：键类型、立体键类型、共轭与环标记。所有真实边保留。BRICS 来自干净分子，这些都是额外结构条件。
- 仅在原子属性空间加高斯噪声，**不破坏或恢复连接，不恢复键，不处理三维坐标**。干净键和拓扑可间接透露芳香性等属性，这属于披露的条件化任务，不能声称无条件恢复结构。
- s 的初始输入也来自 x_t，没有外部原子输入的旁路。s 同时接收时间条件与边界标记。其独立性限定为：给定静态条件后，外部基序的输入数值不经消息进入 s；形式电荷、芳香性本身可能带有环境相关化学信息，不声称物理或化学意义上完全环境无关。
- 每个分子采一个 t∈[1,T]，原子噪声相互独立；T=1000，beta 从 1e-4 线性增加到 0.02，alpha_bar[0]=1。
- 只预测 epsilon，损失为批内所有原子与属性通道的 MSE，没有第二套基序去噪、对比或连接重建损失。恢复公式使用同一个累计 alpha_bar。x0 转换是接口，不是另一个训练目标。
- 门控是每层每条有向跨边界边上的标量 sigmoid，不做 softmax；按入边数量聚合，**不按门控和归一化**，保留抑制消息的效果。它调节信息流，不等于化学因果贡献。
- 微调加载 encoder 的全部权重，由性质损失更新整个 encoder 和新预测头；噪声头冻结且不进入预测。下游显式用 t=0，该状态不要求反向采样。
- 训练 MSE 的大分子目标权重更高；边界/内部误差分开记录，空组为 null。该日志不改变预训练目标。

## 运行

工作目录为项目根目录。当前 `polyolefin_ml` 环境已满足依赖；config.json 记录 Python、Torch、PyG、RDKit、OGB 等版本，以及本模块和复用评价依赖的代码哈希。

```powershell
$py = 'C:/Users/12775/miniconda3/envs/polyolefin_ml/python.exe'
& $py -m unittest motif.brics_atom_diffusion.test_interfaces -v

# 预训练语料 CSV 必须有 smiles 列；请指定实际选定的无标签语料。
# 没有 --valid-csv 时，先按规范 SMILES 去重，再按 seed 划分 10% 无标签验证分子。
& $py -m motif.brics_atom_diffusion.pretrain --smiles-csv path/to/pretrain.csv --output-dir motif/brics_atom_diffusion/runs/example_pretrain --device cuda:0

# 先只做官方 scaffold 验证集评价。架构参数须与预训练检查点一致。
& $py -m motif.brics_atom_diffusion.finetune --dataset esol --pretrained motif/brics_atom_diffusion/runs/example_pretrain/best.pt --epochs 200 --patience 30 --output-dir motif/brics_atom_diffusion/runs/example_esol_pretrained --device cuda:0

# 同架构从头训练对照；仍不读取测试指标。
& $py -m motif.brics_atom_diffusion.finetune --dataset esol --epochs 200 --patience 30 --output-dir motif/brics_atom_diffusion/runs/example_esol_scratch --device cuda:0
```

`--seed` 指定单次训练种子；多个种子使用不同输出目录。微调复用现有 OGB 格式数据读取、官方划分、仅训练集拟合的目标标准化、缺失标签掩码和评价函数，不沿用旧模型权重。测试默认关闭；协议冻结后显式 `--evaluate-test` 才在验证选模完成后评价一次。

预训练的训练/验证规范 SMILES 都保存到配置和检查点，显式预训练 train/valid CSV 间的重复分子被拒绝。微调会检查两部分预训练语料是否包含完整下游 valid/test 中的规范分子，默认有重叠就拒绝；只有明确指定 `--allow-pretrain-overlap` 才允许并记录该暴露，不能把它解释为严格不重叠实验。正式预训练前还应冻结语料来源、规模和对照协议，本次未自动启动正式语料训练。

### 中断恢复

每轮保存 best.pt、latest.pt、history.json 和 status.json。latest.pt 包含优化器、模型、训练数据加载器 RNG、训练噪声 RNG、Torch CPU/CUDA RNG、历史与早停计数。

对未完成运行，使用**相同原命令**加 `--resume` 恢复。源码、数据、配置需完全一致；不能借 resume 修改 epoch 上限、超参数或迁移其他运行的检查点。已完成目录拒绝重复评价或覆盖。进程强制退出留下的 `.lock` 不会被自动删除，先确认原进程已经结束再处理锁，或使用新的输出目录。

### 接口与控制

`--single-state`、`--uniform-boundary`、`--no-motif-prior` 提供初步结构控制开关；会改变计算路径或参数量，**尚不能视为容量匹配消融**。关闭基序先验时，所有边按普通边处理，边界标记归零。连接位点打乱消融尚未实现。

```python
import torch
from motif.brics_atom_diffusion.model import AtomDiffusionModel, Config
from motif.brics_atom_diffusion.training import load_checkpoint

saved = load_checkpoint('path/to/finetune/best.pt')
model = AtomDiffusionModel(Config(**saved['config']), saved['num_tasks'])
model.load_state_dict(saved['model'], strict=True)
model.eval()
with torch.no_grad():
    prediction, states = model(batch, return_states=True)
    if saved['target_scaler'] is not None:
        stats = saved['target_scaler']
        prediction = prediction * stats['scale'] + stats['center']
# states['internal'/'context'/'atoms'] 均逐原子对齐。
# states['cross_edge_index'] 与 states['gates'][layer] 逐条边对齐。
```

batch 使用 data.py 的 AtomData 经 PyG Batch/DataLoader 合批。旧 MotiL 或 context_denoising 检查点不能加载为本模型。

## 已完成的验收（非性能结论）

13 项测试全部通过：受损输入敏感性及梯度、时间条件、无干净特征/编号旁路、内部隔离与上下文响应、原子/边重排、合批与端点、门控对齐、uniform 门控严格等于 1 且保留真实边、已知噪声精确恢复、每分子时间步、单原子/无边及控制开关、两阶段梯度、干净预测不采样。门控消融命令与边界见 [ABLATIONS.md](ABLATIONS.md)。

`runs/smoke_20261029`：

- pretrain：语料来自 ESOL 官方训练集合前 96 个分子，规范去重后划分，仅取 train/valid 各 16 个，2 轮；没有用 ESOL 验证/测试分子做预训练。
- esol_pretrained / esol_scratch：每个官方 split 前 32 个样本，2 轮，仅验证指标。
- bbbp_scratch：每个官方 split 前 128 个样本，1 轮，仅验证指标。
- resume_probe：模拟第 1 轮存盘后中断，按原配置恢复至第 2 轮。模型权重、训练历史及 RNG 状态与连续预训练逐值一致。
- acceptance.json：三份微调最佳权重严格重载，复现保存的验证指标；没有产生测试预测，也不据此判断预训练增益。

这些小样本运行确认计算链可执行，不代表新框架已经达到旧 ESOL 指标或证明扩散/基序先验有益。下一步应先确定正式无标签语料和预训练预算，再建立同架构从头训练与预训练后微调的匹配比较。
