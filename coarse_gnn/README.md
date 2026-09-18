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

## Near-Fine/Far-Coarse packed 执行

新层级分支使用 `HierarchyCache` 在 CPU 离线保存规范拓扑和默认 adaptive context，再由 `pack_hierarchy()` 只为当前 batch 拼接全局索引。`PackedHierarchyPlan.pin_memory().to(device)` 可将 Atom→L1、L1→L2、L2→L3、跨尺度 context 和图级 segment 一次复制到 GPU。GPU 前向没有按分子、父区域或 query 的 Python 循环：L1 pooling、两级父块内部 GINE、segmented attention 和 graph readout 均为 packed tensor 运算。

```python
from coarse_gnn import HierarchicalPredictor, HierarchyNetworkConfig, pack_hierarchy

plan = pack_hierarchy(topologies, padded_nodes=batch.node_features.size(1))
plan = plan.pin_memory().to(device)
model = HierarchicalPredictor(HierarchyNetworkConfig(
    input_dim=encoder.config.hidden_dim,
    base_dim=base_embedding.size(1),
    hidden_dim=128,
)).to(device)
output = model(h2, base_embedding, base_prediction, plan)
```

`pack_hierarchy(..., full_l1=True)` 仅替换 context plan，可在完全相同的网络参数和执行路径下做全 L1 消融。正式 DataLoader 可用 `AtomCountBucketBatchSampler` 按原子数做 sortish batching，降低 dense diffusion 的平方级 padding 浪费。冻结 backbone 的阶段还可将 `h2`、`base_embedding` 和 `base_prediction` 作为数据特征离线保存；联合微调时再恢复在线编码。

RTX 5070 Laptop 上使用现有 4 层/128 维 checkpoint、256 个真实 OGB 分子、BF16 forward+backward 做了交错基准；每种顺序完整预热后重复 3 轮。random/bucket 在同一轮逐 batch 交替执行，以降低功耗状态和测试顺序偏差。

| batch | random samples/s | bucket samples/s | bucket 变化 | padding ratio random→bucket |
|---:|---:|---:|---:|---:|
| 8 | 38.70 | 40.35 | +4.3% | 1.30→1.04 |
| 16 | 71.10 | 72.42 | +1.8% | 1.39→1.04 |
| 32 | 138.26 | 146.23 | +5.8% | 1.49→1.08 |

最坏 batch 的峰值显存约为 210/396/766 MiB；因为两种顺序最终都包含同一个最大分子，峰值几乎不变。交错区间的整卡利用率约为 5.3%/4.3%/9.5%，说明该小分子样本尚未喂满 GPU，不能只凭 packed 结构宣称高利用率。可用 `python -m scripts.validation.benchmark_hierarchy_gpu` 在目标机器和正式数据分布上复现；默认写出未纳入 Git 的 `outputs/hierarchy_gpu_benchmark.json`。

冻结 Base 的正式编排入口为 `scripts.experiments.run_hierarchy_lipo`。`base` 直接读取既有 `pretrained_frozen` 各 seed 的已选 checkpoint；`full_l1` 与 `adaptive` 共享固定的 (h^{(2)})、(h_{base})、Base 预测、初始化和验证协议，只训练新分支。首次阶段不计算测试集指标。

```powershell
conda run --no-capture-output -n polyolefin_ml python -u -m scripts.experiments.run_hierarchy_lipo --action prepare --seeds 0 1 --batch-size 32
conda run --no-capture-output -n polyolefin_ml python -u -m scripts.experiments.run_hierarchy_lipo --action train --seeds 0 1 --variants full_l1 adaptive --batch-size 32
conda run --no-capture-output -n polyolefin_ml python -m scripts.experiments.run_hierarchy_lipo --action summarize --seeds 0 1
```

训练使用按原子数 sortish bucket、原训练器的断点恢复、BF16、验证集 checkpoint selection 和 patience。全数据特征与 hierarchy/context 只在 `prepare` 首次生成，之后命中 `outputs/hierarchy_lipo_frozen` 下的缓存。

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

## 拓扑缓存与预计算

`TopologyCache` 保存 CPU 上的离散拓扑，包括 `atom_order`、反向编号映射、`centers`、`owner`、`cores`、`contexts`、`context_edges`、区域边与池化索引、`coarse_edges`，以及 `boundary_edge_ids` / `boundary_groups` 边界映射。`input_edge_ids` 同时保存原始边存储位置到规范物理边的映射，前向不再用字典重建它。

```python
from coarse_gnn import TopologyCache
from coarse_gnn.diffusion_adapter import DiffusionCoarseModel, precompute_batch_topologies

cache = TopologyCache("outputs/topology_cache", max_memory_entries=128)

# 可在构造模型前遍历 CPU DataLoader；不读取性质标签、不运行编码器。
# 如修改粗化参数，应将同一个 CoarseningConfig 传给预计算和模型。
# for batch in loader:
#     precompute_batch_topologies(batch, cache)

model = DiffusionCoarseModel.from_checkpoint(
    "diffusion/outputs/ogb_clean/encoder.pt", topology_cache=cache,
    freeze_encoder=False,
)
# 每个 epoch 照常运行 model(batch)，自动命中已有拓扑。
# output = model(batch)
print(cache.stats)  # hits: 内存命中；disk_hits: 磁盘命中；misses: 新构建次数
```

也支持显式传入预计算对象，便于数据管线自己管理：

```python
# batch 是已有 MoleculeBatch；预计算不运行神经网络。
topologies = model.precompute_topologies(batch)
output = model(batch, topologies=topologies)
```

显式列表必须与当前 batch 的分子顺序对应。模型会核对输入指纹，误传其他分子、其他编号或其他配置的拓扑会报错。通用接口对应 `predictor.prepare_topology(N, edge_index, node_labels=..., edge_labels=...)` 和 `predictor(..., topology=topology)`；自动缓存通过 `CoarseGraphPredictor(..., topology_cache=cache)` 开启。

缓存规则：

- 指纹包含节点数、精确的输入边排列、原始离散原子/键属性、可选 legacy `node_ids`、全部 `CoarseningConfig` 参数、拓扑实现版本和 igraph 版本。它是原始输入的内容哈希，计算为 `O(N+E)`，不需要先做 canonicalization。
- 节点重编号、边重排或图增强会生成独立条目，避免复用旧 `atom_order` / `edge_positions`。这不是跨同构图共享的缓存；固定数据集输入顺序可最大化命中率。模型的重编号不变性仍由原来的规范化保证。
- 不保存扩散节点向量、训练标签、消息边特征、GNN 输出或计算图；更换网络权重、增强开关、读出方式和微调编码器均可复用同一拓扑。连续边属性在每次前向重新聚合并保留梯度。
- 未传缓存时沿用即时构建。`TopologyCache()` 仅缓存内存；传目录后持久化为张量和基础类型组成的 `.pt` 文件，以 `weights_only=True` 读取。默认最多保留 128 张图的内存条目，超出后按最近使用顺序淘汰，磁盘文件保留；设为 0 可仅用磁盘。`clear_memory()` 不删除磁盘文件。
- 文件采用临时文件加原子替换写入；不同进程可以共用目录，各自持有内存缓存。同时首次处理同一张图时可能重复计算，但不会读取到半写入文件。损坏文件明确报错并提示重建。
- 返回的拓扑对象应视为只读。修改粗化算法、字段格式或索引语义时必须递增 `topology.py` 中的 `TOPOLOGY_VERSION`；旧条目仍保留，但不会命中。默认 `outputs/topology_cache/` 可重建，因此不纳入 Git。

CLI：`python -m scripts.data.precompute_topology --input molecules.csv --cache-dir outputs/topology_cache`。可用 `--smiles-column` 指定 CSV 列；图 JSONL 沿用原始数据格式。`--radius`、`--center-fraction`、`--max-residual-size` 应与模型配置一致。再次运行同一命令会读取已有文件；输出报告中 `misses=0` 表示本次未重新构建。

缓存不消除输入哈希、磁盘 I/O、CPU/GPU 索引传输、分子适配器的稠密边检查和 GNN 计算；实际训练加速幅度仍需在选定数据集上测量。

缓存验收：22 项测试通过（原有 16 项、新增 6 项）。新增测试覆盖磁盘全字段还原、命中后禁止调用 Bliss/BFS、参数/属性/编号变化失效、缓存与即时构建的预测及梯度一致、batch 重排、重编号不变性和连续编码器微调。使用现有预训练权重完成 CPU 预计算后在 CUDA 上微调一步，记录为 4 次磁盘命中、8 次内存命中、0 次重建，见 [运行报告](../outputs/coarse_demo/topology_cache_cuda.json)。该报告使用合成目标验证梯度，不代表下游训练结果。

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
- 当前规范化与拓扑在 CPU 上构建，启用缓存后只在未命中时计算。旧 `CoarseGraphPredictor` 仍是区域逐个编码、批次内粗图逐图处理；新 `HierarchicalPredictor` 已使用 packed batch。首次构建时 Bliss 的最坏时间复杂度是指数级；旧算法其后的中心 BFS 约为 `O(K(N+E))`，上下文边筛选约为 `O(K_final E)`，残余检查另有遍历开销。
- 扩散适配器沿用原编码器的稠密分子边矩阵，因此整体仍有原图二次规模的内存开销。通用粗图预测器不要求稠密矩阵。
- 当前只支持图级预测，不包含粗图向原节点回传或节点级预测头。通用粗图模块不限定材料种类，但现有预训练编码器及其适配器仍是分子图模型。
- 尚未进行正式下游训练、长程性能验证或创新性验证。

## 检查

```powershell
conda run --no-capture-output -n polyolefin_ml python -m unittest discover -s tests -v
conda run --no-capture-output -n polyolefin_ml python -m examples.run_coarse_demo --backward
conda run --no-capture-output -n polyolefin_ml python -m examples.run_coarse_demo --variant base --backward --finetune-encoder --device cuda --output outputs/coarse_demo/canonical_base_finetune_cuda.json
conda run --no-capture-output -n polyolefin_ml python -m scripts.validation.audit_method --permutations 100
conda run --no-capture-output -n polyolefin_ml python -m scripts.validation.audit_method --permutations 100 --variant base --output outputs/method_audit/canonical_base.json
```

正式方法测试不传 persistent ID，每次重排重新运行编码器和完整粗化；断言规范结构完全相同、粗节点与边数一致、预测绝对误差不超过 `1e-6`。增强项测试覆盖 32 种开关组合；同时保留主归属、边去重、上下文、冻结/微调梯度等工程测试。

`outputs/method_audit/current.json` 是修复前记录；`canonical.json` 和 `canonical_base.json` 为修复后的端到端审计。`--legacy-coarsening` 可复现旧的编号敏感性，此时审计失败并返回非零退出码是预期结果。所有预测仍来自未训练下游权重，这些测试不代表真实任务精度或长程能力已得到验证。

规范化依据：[igraph/Bliss 官方说明](https://igraph.org/c/html/develop/igraph-Isomorphism.html)。
