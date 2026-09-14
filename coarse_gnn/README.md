# 粗粒度模型框架

## 已实现的流程

1. 扩散适配器在完整干净分子图、时间步 0 调用 `encode_nodes()`，默认冻结编码器并关闭其 dropout。通用预测器也可直接接收其他编码器的节点表示。
2. 第一中心优先选最大度节点；以后选择到最近已有中心距离最大的未覆盖节点。平局优先最大度，再按节点 ID。初始中心预算为 `ceil(center_fraction * N)`，已全覆盖时提前停止。
3. 每个主中心取原图上 `radius` 跳以内的诱导子图作为上下文，默认四跳。
4. 在未被任何上下文覆盖的节点诱导子图中求连通分量。若某分量超过 `max_residual_size`，在大型残余中的最远节点补中心并重新检查；每次至少覆盖一个新节点，循环可终止。
5. 最终以最近中心确定被覆盖节点的互斥主归属，平局归属于较早选择的中心。小型残余连通分量单独形成粗节点。所有节点恰好归属于一个粗节点，且主归属是其编码上下文的子集。
6. **所有归属确定后**才开始区域编码。区域共享同一套参数；在整个上下文内运行带边特征 GIN，只池化属于该粗节点的节点。单节点也走相同编码器，保留自身变换。残余的上下文仅为其自身，不额外扩展。
7. 按主归属聚合跨区域原始边。一条无向原始边只计一次，粗边特征为 `[边数, 原始边属性的 sum/mean]`。区域内部边不形成粗图自环；网络具有自身状态更新。
8. 可将主区域大小的 `log1p` 投影加到粗节点表示，再运行带边特征 GIN。图级读出可选 mean、sum、按主区域大小加权的 mean，最后输出预测值或 logits。

默认区域 GNN 为 2 层、粗图 GNN 为 3 层、隐藏维度 128。实现采用边投影、双向求和消息、自身更新、残差与 LayerNorm，没有新增 Gate。两个 GNN 的参数互不共享，各区域的区域 GNN 参数共享。

## 通用图接口

```python
import torch
from coarse_gnn import CoarseGraphPredictor, CoarseningConfig, NetworkConfig

model = CoarseGraphPredictor(
    NetworkConfig(input_dim=64, hidden_dim=128, edge_dim=0, output_dim=1),
    CoarseningConfig(radius=4, center_fraction=0.1, max_residual_size=4),
)
nodes = torch.randn(30, 64)  # 任意编码器的输出，可保持梯度
edges = torch.stack((torch.arange(29), torch.arange(1, 30)))
output = model(nodes, edges)
print(output.prediction.shape)  # [1]
print(output.topology.stats)
print(output.topology.owner)    # 每个原始节点的最终粗节点归属
```

输入支持无向简单图：

- `node_embeddings`: `[N, input_dim]`，有限浮点数，非空。
- `edge_index`: `[2, E]` 的 long 张量。可每条无向边存一次，也可存正反两个方向；两个方向的属性必须一致。拒绝自环、同向重复边和多重边。
- `edge_attr`: `[E, edge_dim]`，可微分的连续属性。`edge_dim=0` 时可省略。类别需事先 one-hot 或嵌入，不能直接平均类别编号。
- `node_ids`: 可选 `[N]` 唯一 long ID，用于中心和残余排序的平局处理。

输出 `GraphOutput` 包含预测、图表示、粗图传播前后的区域表示、粗边属性、完整拓扑及统计。结构张量在 CPU；预测和表示在模型所在设备。离散选中心不可微，特征编码、池化、边属性聚合和预测可反向传播。

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

`from_checkpoint()` 自动读取编码器维度。自定义网络时传入 `NetworkConfig(input_dim=128, edge_dim=4, ...)`，适配器会检查维度；分子边属性为单、双、三、芳香键的四维 one-hot。适配器处理变长批次及 padding，拒绝带 mask 键的加噪图，拓扑始终取干净图。

微调时在创建模型时设置 `freeze_encoder=False`。也可以调用 `set_encoder_frozen(False)`，但切换后应重新建立包含新增可训练参数的优化器。扩散解码头及原有图投影头不参与下游训练。

输出默认不施加 sigmoid/softmax：回归可用 MSE，二分类可用 `BCEWithLogitsLoss`，多分类设置 `output_dim=类别数` 并用 `CrossEntropyLoss`。本框架不定义尚未选定的数据划分、标签归一化或训练目标。

## 诊断与消融

`output.topology.stats` 记录原节点/边数量、主区域和残余数量、补中心数、最终粗节点和粗边数、最大主区域大小，以及：

- `coarse_node_ratio`：最终粗节点数 / 原节点数，包含残余补充。
- `context_occurrence_ratio`：所有最终编码上下文中节点出现次数 / 原节点数，包含残余编码范围；完整覆盖时至少为 1。

不强制把重复出现比例限制为 1.25；它目前只是待实验检验的参考目标。固定跳数在高分支图上也可能得到很大的区域，节点数和拓扑距离的缩短程度需要实测。

`region_layers=0` 可检查去掉区域消息传递的版本（保留输入投影和池化）；`coarse_layers=0` 可检查去掉粗图传播的版本；`graph_pool`、`region_pool`、`edge_reduce`、`use_size_feature` 可独立设置。

## 当前边界

- **默认划分依赖节点编号的平局处理，不保证任意重编号下的图同构不变性。** 若调用方有可跟随重排的稳定 ID，可通过 `node_ids` 保持同一图在该重排下的选择一致；这不等于实现了图规范化。分子适配器接受每张图一个 ID 张量的列表，位置对应其有效原子行。
- 残余区域保证节点进入表示，但不保证大幅压缩或消除长程瓶颈；主区域的四跳半径也不限制高分支图的节点数。
- 区域上下文不保证为每个边界主节点提供完整的多层外部邻域。共享全图扩散表示提供已有上下文，区域 GNN 使用其截取到的诱导子图。
- 当前拓扑在 CPU 上通过 BFS 构建，每次前向重新计算，区域逐个编码，批次内粗图逐图处理。中心 BFS 约为 `O(K(N+E))`，上下文边筛选约为 `O(K_final E)`，残余检查另有遍历开销；尚未实现静态拓扑缓存或区域并行批处理。
- 扩散适配器沿用原编码器的稠密分子边矩阵，因此整体仍有原图二次规模的内存开销。通用粗图预测器不要求稠密矩阵。
- 当前只支持图级预测，不包含粗图向原节点回传或节点级预测头。通用粗图模块不限定材料种类，但现有预训练编码器及其适配器仍是分子图模型。
- 尚未进行正式下游训练、长程性能验证或创新性验证。

## 检查

```powershell
conda run --no-capture-output -n polyolefin_ml python -m unittest discover -s tests -v
conda run --no-capture-output -n polyolefin_ml python run_coarse_demo.py --backward
conda run --no-capture-output -n polyolefin_ml python run_coarse_demo.py --backward --finetune-encoder --device cuda --output outputs/coarse_demo/finetune_cuda.json
```

测试覆盖主归属完整性、上下文包含主节点、残余补中心、断开分量、大半径、无向边去重、粗边属性、只池化主区域、上下文影响、稳定 ID 重排、批次 padding、冻结/微调梯度与优化器更新。
