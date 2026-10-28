# 第一阶段：基于扩散加噪的基序上下文去噪辅助学习

独立实验实现，不调用历史原子／键扩散路径。目标是检验性质预测收益，尚无正式性能结论。

## 实现

- `data.py`：RDKit BRICS 找断键，删除这些边后的连通分量作为非重叠基序。所有原子恰好归属一个基序；不添加虚拟原子。原子 GNN 始终保留完整分子图。跨基序键按双向边保存，记录两端原子和 OGB 键类别。无可断键分子允许只有一个基序；盐等不连通组分各自保留。
- `model.py`：OGB 原子／键特征 → 完整图残差 GINE → 基序均值池化 → 无可学习仿射参数的 LayerNorm 得到 q。原子读出采用求和池化，基序读出采用均值池化。关系感知消息采用入边均值聚合与门控残差融合。
- 干净关系适配器接收连接位点原子隐表示及键类别编码；去噪适配器仅接收静态键类别编码。边索引表达归属／连接结构，缺失关系不产生边。两条分支共享同一组消息传递层、时间条件和受损标记编码；干净状态固定 t=0、标记=0。
- 每个分子随机选 ceil(K × corruption_rate) 个基序，至少一个；每个分子采一个 t，所选基序共享 t，噪声独立。未选基序使用 q、t=0、标记=0。余弦 alpha_bar 日程；固定噪声默认 t=50/100。mask 模式把目标输入完全置零。
- 去噪预测干净 q，损失为所选基序 squared-L2 的均值，目标停止梯度。它不是按隐藏维度取平均的 MSE，因此默认权重 0.01 需要用验证集选择。
- 推理只运行干净分支；无需噪声、去噪头或反向采样。完整原子编码已传播目标信息，因此不宣称严格的缺失基序推断。

## 实验开关

| variant | 性质分支 | 辅助任务 |
|---|---|---|
| atom | 原子读出 | 无 |
| motif | 原子＋普通 q 读出 | 无 |
| context | 原子＋上下文增强 q | 无 |
| fixed | 同 context | 固定时间步 |
| random | 同 context | 随机时间步 |
| no_communication | 同 context | 随机时间步，仅去噪分支关闭跨基序消息 |
| mask | 同 context | 完全遮蔽所选 q |
| shuffle | 同 context | 随机时间步，仅去噪边在每个分子内随机重连 |

shuffle 使用分子内基序节点随机置换来改变边与基序内容的对应，保留边数及边特征。小图／对称图可能不发生有效改变，不能把这个对照单独当作上下文证据。单基序样本也不提供基序间上下文证据，需结合分解审计统计解释结果。

## 数据与已有基线

复用 `model/benchmark.py` 的数据读取、官方 scaffold split、训练集目标标准化、缺失标签掩码及评价函数。回归 MSE；分类 BCEWithLogits，正类权重仅由训练集计算。

已检查历史 `model/results_formal/01_local/downstream_summary.csv`：ESOL GINE 五种子 RMSE 0.94599 ± 0.03834。这是历史参考，并非本方法对照。当前模型的 MLP 激活与预测头设置不同；公平比较应使用本入口重新训练的 atom/motif/context 配置。

首次全体消融运行使用相同初始化、数据批次顺序、预测主干与训练超参数；噪声使用独立 RNG，避免改变干净分支的 dropout 随机序列。不同辅助任务仍有不同计算耗时，记录实际时间。验证集早停意味着实际 epoch 数可不同；可将 patience 设为 epochs 来固定训练轮数。GPU 当前允许非确定性聚合，配置文件显式记录，多种子用于评估波动。

## 运行

工作目录为项目根目录。现有 `polyolefin_ml` 环境具备 torch、PyG、OGB、RDKit 等依赖，无需新装环境。

```powershell
$py = 'C:/Users/12775/miniconda3/envs/polyolefin_ml/python.exe'
& $py -m unittest motif.context_denoising.test_method -v

# 小样本验收；这些指标不能用作方法性能结论。
& $py -m motif.context_denoising.train --dataset esol --seeds 0 --epochs 2 --patience 2 --hidden-dim 16 --atom-layers 2 --batch-size 16 --limit-per-split 32 --device cpu --output-dir motif/context_denoising/runs/my_smoke

# 正式运行的第一步：先比较前三个架构，仅使用验证集。
& $py -m motif.context_denoising.train --dataset esol --variants atom motif context --seeds 0 1 2 3 4 --output-dir motif/context_denoising/runs/stage1_architecture

# 再比较辅助任务。冻结超参数后，加 --evaluate-test 才读取测试指标。
& $py -m motif.context_denoising.train --dataset esol --variants context fixed random no_communication mask shuffle --seeds 0 1 2 3 4 --output-dir motif/context_denoising/runs/stage1_auxiliary
```

输出目录不可覆盖已有 run。保存分解审计、配置、数据／代码／split 哈希、每轮性质与去噪损失、q 范数与批内维度方差、最佳权重及标准化参数、各随机种子指标与均值／标准差。q 范数受 LayerNorm 控制，维度方差为批内诊断，不能代替全数据表示分析。测试默认关闭；启用后每个 run 在验证选模结束时只评价一次。

## 重载预测

```python
import torch
from motif.context_denoising.model import Config, MotifContextModel

checkpoint = torch.load('path/to/best.pt', map_location='cpu', weights_only=True)
model = MotifContextModel(checkpoint['num_tasks'], Config(**checkpoint['config']))
model.load_state_dict(checkpoint['state_dict'])
model.eval()
with torch.no_grad():
    prediction = model(batch, variant=checkpoint['variant'])
    scaler = checkpoint['target_scaler']
    if scaler is not None:
        prediction = prediction * scaler['scale'] + scaler['center']
```

batch 使用 `data.py` 的 MotifData 经 PyG Batch/DataLoader 合批。检查点保存完整模块便于恢复，推理 forward 不调用辅助路径。

## 已完成验收

`test_method.py` 验证归属／连接位点合批、单原子无边、干净分支一致性、部分加噪、每分子时间步、共享主干梯度、去噪禁止通信与分批预测一致性。

`runs/smoke_20261028`：ESOL 每个官方 split 前 32 个样本，全部八种配置各训练两轮，包含最佳权重与一次最终测试的流程验收。共 96 个分子，其中 45 个只有一个基序；不能据此比较方法性能。正式实验尚未运行。

八份 ESOL 检查点已重载，并在单分子输入上复现对应的保存预测（容差 1e-5）。`runs/smoke_classification_bbbp_20261028` 完成 BBBP 每 split 前 128 个样本、random/no_communication 各一轮的分类训练／选模／测试验收。此前 BACE 前 64 个验证样本仅一个类别，无法评价 ROC-AUC，该未完成运行保留于 `runs/smoke_classification_20261028`；入口已增加不可评价时的明确提示。
