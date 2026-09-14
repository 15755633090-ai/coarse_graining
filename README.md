# 基于粗粒度化的扩散 GNN

本项目面向普通分子数据集的图级性质预测。审计中的合成链图和分子图仅用于检查重编号不变性，不限定实际数据集的分子结构。

模型框架：完整原图扩散编码 → 原始离散属性图规范化 → 分散中心与四跳上下文 → 互斥主归属与残余补充 → 区域 GNN → 粗图 GNN → 图级预测。

规范化使用 igraph/Bliss：原子颜色包含五类原始离散属性，键类型通过辅助节点颜色保留。128 维扩散表示不参与规范化。默认保留增强版配置，也可用 `--variant base` 运行无显式区域/粗边特征、无区域大小特征、mean readout 的基础版。

新增模块从现有扩散编码器的 `encode_nodes()` 获取节点表示；原始编码器及内部消息门控保留。没有额外的重要节点 Gate、回溯或子图筛选模块。

## 目录

```text
coarse_graining/
├── coarse_gnn/
│   ├── config.py             # 粗化规则和网络参数
│   ├── canonical.py          # 原子属性与键类型的 Bliss 规范化
│   ├── cache.py              # 拓扑内存缓存与磁盘持久化
│   ├── topology.py           # 中心、归属、残余、上下文与粗边
│   ├── layers.py             # 纯 PyTorch 的带边特征 GIN
│   ├── model.py              # 接收任意节点表示的通用粗图预测器
│   ├── diffusion_adapter.py  # 与现有扩散编码器、分子批次连接
│   └── README.md             # 接口、算法细节、限制与训练示例
├── diffusion/                # 原扩散预训练模块与权重
├── tests/test_coarse_gnn.py   # 拓扑、表示、批次、梯度检查
├── run_coarse_demo.py         # 现有权重加载与前向/反向演示
├── audit_method.py            # 完整重编号不变性审计
├── precompute_topology.py     # 数据集粗化拓扑离线预计算，无需编码器权重
├── requirements.txt           # 完整依赖，包括 igraph
└── outputs/coarse_demo/      # 演示输出，不覆盖预训练权重
```

## 运行

在本项目根目录，使用已有 `polyolefin_ml` 环境。网络使用纯 PyTorch，规范化另需 `igraph==0.11.9`（本机已安装），不要求 PyTorch Geometric；SMILES 输入另需 RDKit（已有环境提供）。新环境先执行 `python -m pip install -r requirements.txt`。`--no-capture-output` 避免 Windows 下 conda 对中文路径输出的转码问题。

```powershell
conda run --no-capture-output -n polyolefin_ml python run_coarse_demo.py --backward
conda run --no-capture-output -n polyolefin_ml python run_coarse_demo.py --variant base --backward --output outputs/coarse_demo/base_report.json
conda run --no-capture-output -n polyolefin_ml python run_coarse_demo.py --smiles 'CCO' 'CC(=O)O' 'c1ccccc1' '[Na+].[Cl-]'
conda run --no-capture-output -n polyolefin_ml python -m unittest discover -s tests -v
conda run --no-capture-output -n polyolefin_ml python audit_method.py --permutations 100
```

默认加载 `diffusion/outputs/ogb_clean/encoder.pt`，在时间步 0 冻结编码器。无 SMILES 参数时使用 1、24、100 节点的合成链图，输出每张图的规范编号、原始编号映射、归属、上下文、粗边和覆盖统计到 `outputs/coarse_demo/canonical_report.json`。`--backward` 使用合成标签执行一次优化器更新，仅用于验证梯度链路。加 `--finetune-encoder` 可检查编码器微调；加 `--device cuda` 使用 GPU。旧的 `report.json` 等演示输出是规范化修复前的历史记录。

**下游区域网络、粗图网络和预测头尚未在真实性质标签上训练；演示预测值不能作为性质预测结果。** 当前没有启动正式训练，也未选择下游数据集或目标性质。

## 预计算粗化拓扑

可以先为普通分子数据集建立磁盘缓存，再在训练中复用。支持 SMILES CSV（默认列名 `smiles`）、`.smi`、`.txt` 和现有分子图 JSONL；预计算不加载编码器权重。

```powershell
conda run --no-capture-output -n polyolefin_ml python precompute_topology.py --input molecules.csv --cache-dir outputs/topology_cache
```

用分子示例检查预计算和模型读取：

```powershell
conda run --no-capture-output -n polyolefin_ml python precompute_topology.py --smiles 'CCO' 'CC(=O)O' 'c1ccccc1' --cache-dir outputs/topology_cache
conda run --no-capture-output -n polyolefin_ml python run_coarse_demo.py --smiles 'CCO' 'CC(=O)O' 'c1ccccc1' --topology-cache outputs/topology_cache --output outputs/coarse_demo/cache_report.json
```

训练时向 `DiffusionCoarseModel.from_checkpoint(..., topology_cache=TopologyCache("outputs/topology_cache"))` 传入缓存即可。相同输入首次构建，后续从内存或磁盘读取；命中后不再执行 canonicalization、BFS、区域与粗边构造。图结构、原始离散属性、输入编号或粗化参数变化时生成新条目。具体接口与失效规则见 [缓存说明](coarse_gnn/README.md#拓扑缓存与预计算)。

通用图接口和训练用法见 [粗粒度模块说明](coarse_gnn/README.md)。原扩散训练与编码用法见 [扩散模块说明](diffusion/README.md)，原编码器核验见 [编码器核验报告](diffusion/ENCODER_AUDIT.md)。
