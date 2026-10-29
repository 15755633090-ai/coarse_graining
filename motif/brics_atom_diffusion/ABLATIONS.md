# 门控与结构消融入口

默认模型已经包含条件边界门控。每层每条有向跨基序真实边，根据源/目标的 s、h 以及该边键特征计算标量 sigmoid 门控，再乘到跨边界消息。预训练恢复损失和微调性质损失均能训练门控；对应梯度接口已测试。

| 配置 | 参数 | 跨边界处理 |
|---|---|---|
| 条件门控（默认） | 无需额外开关 | 学习 g∈(0,1)，逐条调节消息 |
| 统一聚合 | --uniform-boundary | 保留全部真实边，固定 g=1，不创建门控 MLP |
| 单状态控制 | --single-state | 不再保留独立 s 路径；其余关系逻辑保留 |
| 整套基序关系机制移除 | --no-motif-prior | 所有边普通聚合，s/h 均全图传播，边界标记/门控失效 |

统一聚合会减少参数量，**不是严格容量匹配对照**，实验报告需要同时列出 optimized 参数数。不同开关可组合，但第一轮建议每次只改一个因素。

--no-motif-prior 不是“只移除一个 BRICS 标记”的单因素消融，而是整套关系机制移除对照；其无效边界/跨消息/门控参数会冻结，不计入优化器参数量。

## 门控对照应成对训练

以同一无标签语料、语料划分、种子、预算分别预训练两组，再各自微调。当前严格迁移检查要求预训练与微调的架构配置一致，不能把带门控检查点直接传给 --uniform-boundary 的模型。

```powershell
$py = 'C:/Users/12775/miniconda3/envs/polyolefin_ml/python.exe'
$corpus = 'path/to/pretrain.csv'

# 默认条件门控
& $py -m motif.brics_atom_diffusion.pretrain --smiles-csv $corpus --seed 0 --output-dir motif/brics_atom_diffusion/runs/gated_pretrain_seed0
& $py -m motif.brics_atom_diffusion.finetune --dataset esol --seed 0 --pretrained motif/brics_atom_diffusion/runs/gated_pretrain_seed0/best.pt --output-dir motif/brics_atom_diffusion/runs/gated_esol_seed0

# 仅改变边界门控因素：固定 g=1
& $py -m motif.brics_atom_diffusion.pretrain --smiles-csv $corpus --seed 0 --uniform-boundary --output-dir motif/brics_atom_diffusion/runs/uniform_pretrain_seed0
& $py -m motif.brics_atom_diffusion.finetune --dataset esol --seed 0 --uniform-boundary --pretrained motif/brics_atom_diffusion/runs/uniform_pretrain_seed0/best.pt --output-dir motif/brics_atom_diffusion/runs/uniform_esol_seed0
```

两组默认均不评价测试集。固定相同种子也不保证不同架构的所有随机初始化/噪声轨迹完全一致；初始化消耗和训练路径可能不同。应报告多种子配对结果和各组参数量，不以单次训练差值证明机制。

## 当前状态

13 项接口测试包括：门控在 (0,1)、与真实边对齐、预训练/微调梯度能到达门控，以及 uniform 模式所有跨边界权重严格等于 1 且真实边保留。此前端到端验收使用默认门控，属于小样本流程测试。

**尚未运行新框架正式的“门控 vs 统一聚合”性能消融**；尚未实现容量匹配门控控制与真实连接位点打乱实验。旧 context_denoising 的 40 次 ESOL 结果不属于这组消融。
