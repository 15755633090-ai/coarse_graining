# 粗粒度模型框架

## 已实现的流程

1. 扩散适配器在完整干净分子图、时间步 0 调用 `encode_nodes()`，默认冻结编码器并关闭其 dropout。通用预测器也可直接接收其他编码器的节点表示。
2. 使用原始离散属性规范化：五类原子属性组合形成原子颜色；每条物理键转换成辅助节点，其颜色记录键类型。两个颜色空间分离。igraph/Bliss 返回规范顺序后移除辅助节点，恢复原图的化学键，再进行粗化。辅助节点不算作原子，不参与四跳距离或消息传递。规范化不读取扩散表示或模型参数。
3. 第一中心优先选最大度节点；以后选择到最近已有中心距离最大的未覆盖节点。平局优先最大度，再按规范编号。初始中心预算为 `ceil(center_fraction * N)`，已全覆盖时提前停止。
4. 每个主中心取原图上 `radius` 跳以内的诱导子图作为上下文，默认四跳。
5. 在未被任何上下文覆盖的节点诱导子图中求连通分量。若某分量超过 `max_residual_size`，在大型残余中的最远节点补中心并重新检查；每次至少覆盖一个新节点，循环可终止。
6. 最终以最近中心确定被覆盖节点的互斥主归属，平局归属于较早选择的中心。小型残余连通分量单独形成粗节点。所有节点恰好归属于一个粗节点，且主归属是其编码上下文的子集。
7. **所有归属确定后**才开始区域编码。区域共享同一套参数；在整个上下文内运行 GIN，只池化属于该粗节点的节点。单节点也走相同编码器，保留自身变换。残余的上下文仅为其自身，不额外扩展。
8. 按主归属聚合跨区域原始边。一条无向原始边只计一次，粗边存在性由真实原子边决定；边数、原始边属性 sum/mean 可分别启用。区域内部边不形成粗图自环；网络具有自身状态更新。
9. 可将主区域大小的 `log1p` 投影加到粗节点表示，再运行粗图 GIN。图级读出可选 mean、sum、按主区域大小加权的 mean，最后输出预测值或 logits。

默认保留增强配置：区域 GNN 为 2 层、粗图 GNN 为 3 层、隐藏维度 128，所有统计特征开关开启。实现采用双向求和消息、自身更新、残差与 LayerNorm；启用边特征时增加边投影和消息 ReLU，关闭时使用普通 GIN 的邻居求和。没有新增 Gate。两个 GNN 的参数互不共享，各区域的区域 GNN 参数共享。

## 通用图接口

```python
import torch
from coarse_gnn import CoarseGraphPredictor, CoarseningConfig, NetworkConfig

model = CoarseGraphPredictor(
    NetworkConfig(input_dim=64, hidden_dim=128, edge_dim=0, output_dim=1),
    CoarseningConfig(radius=4, center_fraction=0.1, max_residual_size=4),
)
raw_labels = (torch.arange(30) % 3).unsqueeze(1)  # 原始离散节点类别
nodes = torch.nn.functional.one_hot(raw_labels[:, 0], num_classes=64).float()
# 实际任务可用来自置换等变编码器的节点表示替代 nodes，保持梯度。
edges = torch.stack((torch.arange(29), torch.arange(1, 30)))
output = model(nodes, edges, node_labels=raw_labels)
print(output.prediction.shape)  # [1]
print(output.topology.stats)
print(output.topology.owner_input)  # 输入顺序下每个原始节点的粗节点归属
```

输入支持无向简单图：

- `node_embeddings`: `[N, input_dim]`，有限浮点数，非空。
- `edge_index`: `[2, E]` 的 long 张量。可每条无向边存一次，也可存正反两个方向；两个方向的属性必须一致。拒绝自环、同向重复边和多重边。
- `edge_attr`: `[E, edge_dim]`，可微分的连续属性。`edge_dim=0` 时可省略。类别需事先 one-hot 或嵌入，不能直接平均类别编号。
- `node_labels`: 默认必填，原始离散属性组合的 long 张量 `[N, F]` 或 `[N]`。不能把连续隐藏表示强转整数作为标签。无属性图可显式传 `[N, 0]`，但其编码器也必须尊重该图的对称性。
- `edge_labels`: 原始离散边属性的 long 张量 `[E, F]` 或 `[E]`。有显式边属性时必须提供；不依赖消息传递是否启用边特征。分子适配器始终提供键类型，包括 `edge_dim=0` 的基础版。
- `node_ids`: 仅用于 `CoarseningConfig(canonicalize=False)` 的历史对照模式。默认规范化模式拒绝外部 persistent ID，避免误把外部编号当作图属性。

输出 `GraphOutput` 包含预测、图表示、粗图传播前后的区域表示、实际启用的粗边属性、完整拓扑及统计。结构张量在 CPU；预测和表示在模型所在设备。离散规范化和选中心不可微，但索引重排不截断节点/边特征梯度。

**索引约定：** `owner`、`centers`、`cores`、`contexts`、`edges` 默认使用规范原子编号；`atom_order[规范编号] = 输入编号`，`input_to_canonical` 为其逆映射。`owner_input` 和 `centers_input` 可直接回查输入原子。`edge_positions` 始终指向调用者原始 `edge_index` 的条目。`canonical_signature()` 包含规范后的原始属性、归属、上下文、残余标志和粗边计数，用于方法测试。

## 连接预训练权重并训练

以下代码在项目根目录运行：

```python
import torch
from coarse_gnn.diffusion_adapter import DiffusionCoarseModel
from diffusion.bond_diffusion.data import collate_graphs, graph_from_smiles

model = DiffusionCoarseModel.from_checkpoint(
    'diffusion/outputs/ogb_clean/encoder.pt', freeze_encoder=True,
)
batch = collate_graphs([graph_from_smiles('CCO'), graph_from_smiles('CC(=O)O')])
output = model(batch)
assert output.predictions.shape == (2, 1)

# 实际使用时，targets 应由下游数据集提供且与 batch 顺序一致。
# optimizer = torch.optim.AdamW(
#     [p for p in model.parameters() if p.requires_grad], lr=1e-3,
# )
# model.train()
# optimizer.zero_grad(set_to_none=True)
# output = model(batch)
# loss = torch.nn.functional.mse_loss(output.predictions, targets)  # targets: [B, 1]
# loss.backward()
# optimizer.step()
```

`from_checkpoint()` 自动读取编码器维度。自定义网络可传 `NetworkConfig(input_dim=128, edge_dim=4, ...)`；消息边属性为四类键的 one-hot。也支持 `edge_dim=0`，不传显式消息边属性，规范化仍保留原始键类型。适配器处理变长批次及 padding，拒绝带 mask 键的加噪图，拓扑始终取干净图。

微调时在创建模型时设置 `freeze_encoder=False`。也可以调用 `set_encoder_frozen(False)`，但切换后应重新建立包含新增可训练参数的优化器。扩散解码头及原有图投影头不参与下游训练。

输出默认不施加 sigmoid/softmax：回归可用 MSE，二分类可用 `BCEWithLogitsLoss`，多分类设置 `output_dim=类别数` 并用 `CrossEntropyLoss`。本框架不定义尚未选定的数据划分、标签归一化或训练目标。

## 诊断与消融

`output.topology.stats` 记录原节点/边数量、主区域和残余数量、补中心数、最终粗节点和粗边数、最大主区域大小，以及：

- `coarse_node_ratio`：最终粗节点数 / 原节点数，包含残余补充。
- `context_occurrence_ratio`：所有最终编码上下文中节点出现次数 / 原节点数，包含残余编码范围；完整覆盖时至少为 1。

不强制把重复出现比例限制为 1.25；它目前只是待实验检验的参考目标。固定跳数在高分支图上也可能得到很大的区域，节点数和拓扑距离的缩短程度需要实测。

`region_layers=0` 可检查去掉区域消息传递的版本（保留输入投影和池化）；`coarse_layers=0` 可检查去掉粗图传播的版本。

| 配置 | 默认增强版 | `NetworkConfig.base(...)` |
|---|---|---|
| `use_region_edge_features` | true | false |
| `use_coarse_edge_count` | true | false |
| `use_coarse_edge_features` | true | false |
| `use_size_feature` | true | false |
| `graph_pool` | size_weighted_mean | mean |

这些开关相互独立，不改变用于规范化的原始化学图。`edge_dim` 指输入消息边属性的维度，`coarse_edge_dim` 自动根据开启项计算；全部关闭时粗边属性形状为 `[Ec, 0]`，仅二值拓扑参与粗图传播。`edge_reduce` 控制启用的键类型属性如何聚合。

```python
from coarse_gnn import NetworkConfig
base = NetworkConfig.base(input_dim=128, edge_dim=0)
# 仅增加粗边数量：
base_plus_counts = NetworkConfig.base(input_dim=128, edge_dim=0, use_coarse_edge_count=True)
```

## 当前边界

- 默认在带属性规范图上粗化；对称原子映射回输入时可以互换，不要求同一个外部 atom ID 始终当选中心。保证对象是规范化后的带属性图和图级预测，不是输入编号上的唯一中心集合。旧的 `canonicalize=False` 会恢复已知的编号敏感性，仅用于对照。
- 规范化依赖原始属性必须完整，以及输入编码器对这些属性图置换等变。对任意带有额外、未纳入原始属性的节点特征，不能仅靠拓扑规范化承诺预测不变。不同 igraph/Bliss 版本可能产生不同规范顺序，因此固定 `igraph==0.11.9` 和 `sh="fl"`。
- 当前分子适配器使用二维连接关系、已有离散原子属性和键类型。
- 残余区域保证节点进入表示，但不保证大幅压缩或消除长程瓶颈；主区域的四跳半径也不限制高分支图的节点数。
- 区域上下文不保证为每个边界主节点提供完整的多层外部邻域。共享全图扩散表示提供已有上下文，区域 GNN 使用其截取到的诱导子图。
- 当前规范化与拓扑在 CPU 上构建，每次前向重新计算，区域逐个编码，批次内粗图逐图处理。Bliss 的最坏时间复杂度是指数级；在其后，中心 BFS 约为 `O(K(N+E))`，上下文边筛选约为 `O(K_final E)`，残余检查另有遍历开销。尚未实现缓存或区域并行批处理。
- 扩散适配器沿用原编码器的稠密分子边矩阵，因此整体仍有原图二次规模的内存开销。通用粗图预测器不要求稠密矩阵。
- 当前只支持图级预测，不包含粗图向原节点回传或节点级预测头。通用粗图模块不限定材料种类，但现有预训练编码器及其适配器仍是分子图模型。
- 尚未进行正式下游训练、长程性能验证或创新性验证。

## 检查

```powershell
conda run --no-capture-output -n polyolefin_ml python -m unittest discover -s tests -v
conda run --no-capture-output -n polyolefin_ml python run_coarse_demo.py --backward
conda run --no-capture-output -n polyolefin_ml python run_coarse_demo.py --variant base --backward --finetune-encoder --device cuda --output outputs/coarse_demo/canonical_base_finetune_cuda.json
conda run --no-capture-output -n polyolefin_ml python audit_method.py --permutations 100
conda run --no-capture-output -n polyolefin_ml python audit_method.py --permutations 100 --variant base --output outputs/method_audit/canonical_base.json
```

正式方法测试不传 persistent ID，每次重排重新运行编码器和完整粗化；断言规范结构完全相同、粗节点与边数一致、预测绝对误差不超过 `1e-6`。增强项测试覆盖 32 种开关组合；同时保留主归属、边去重、上下文、冻结/微调梯度等工程测试。

`outputs/method_audit/current.json` 是修复前记录；`canonical.json` 和 `canonical_base.json` 为修复后的端到端审计。`--legacy-coarsening` 可复现旧的编号敏感性，此时审计失败并返回非零退出码是预期结果。所有预测仍来自未训练下游权重，这些测试不代表真实任务精度或长程能力已得到验证。

规范化依据：[igraph/Bliss 官方说明](https://igraph.org/c/html/develop/igraph-Isomorphism.html)。
