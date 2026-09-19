# 基于粗粒度化的扩散 GNN

面向普通分子数据集的图级性质预测：完整原图扩散编码 → 分散中心与重叠上下文 → 互斥 core 归属与残余补充 → Region GNN → Coarse GNN → 图级预测。

当前实验采用 **Frozen Region-only / Frozen Base coarse**。整理目录时（2026-09-15）检测到 seed 2/3/4 正在补跑；实时进度以结果目录中的 `progress.json` 和 `result.json` 为准。实验说明见 [Frozen 机制验证](docs/FROZEN_MECHANISM.md)，暂停的 finetune 计划见 [Lipo 正式协议](docs/LIPO_FORMAL.md)。

## 目录

```text
coarse_graining/
├── run_lipo_frozen.py          # 当前 Frozen 实验入口
├── run_lipo_formal.py          # 正式协议接入与共享训练支持
├── frozen_features.py         # 固定编码器特征缓存
├── coarse_gnn/                # 粗化及 Region/Coarse GNN 核心实现
├── diffusion/                 # 原扩散子项目、数据与预训练权重
├── examples/                  # 最小模型演示
├── scripts/
│   ├── data/                  # 拓扑预计算
│   ├── experiments/           # 第一轮核心消融：统一母模型开关与入口
│   ├── validation/            # 方法不变性、数值等价性检查
│   ├── performance/           # 性能分析、断点测速
│   ├── launchers/             # Windows 后台启动脚本
│   └── maintenance/           # 可选打包工具
├── tests/                     # 自动化测试
├── docs/                      # 实验协议、目录索引、上传说明
├── outputs/                   # 演示、审计、性能报告和缓存
└── requirements.txt           # 项目依赖
```

根目录的三个训练文件保留原位置。Frozen 汇总已扩展到 seed 0–4，通过严格的 reporting-only 核验保持原训练协议兼容，并单独记录实际报告源码。其他入口已按用途分类，完整对应关系见 [目录与命令索引](docs/DIRECTORY_LAYOUT.md)。

## 常用入口

下列命令均在本目录运行，使用已有 `polyolefin_ml` 环境。迁移后的 Python 工具使用 `python -m` 启动。

```powershell
# 查看当前训练入口参数，不启动训练
conda run --no-capture-output -n polyolefin_ml python run_lipo_frozen.py --help

# Base 模型演示与一次前向/反向检查
conda run --no-capture-output -n polyolefin_ml python -m examples.run_coarse_demo --variant base --backward --output outputs/coarse_demo/new_smoke.json

# 预计算分子粗化拓扑
conda run --no-capture-output -n polyolefin_ml python -m scripts.data.precompute_topology --smiles 'CCO' 'CC(=O)O' --cache-dir outputs/topology_cache

# 自动化测试
conda run --no-capture-output -n polyolefin_ml python -m unittest discover -s tests -v

# 层级模型训练前诊断（不启动正式训练）
conda run --no-capture-output -n polyolefin_ml python -u -m scripts.experiments.run_hierarchy_lipo --action prepare --seeds 0 1 --batch-size 32 --micro-batch-size 32

# 验证集选择的四组对照：Base 直接复用，训练 Base+Corr / Full-L1 / Adaptive
conda run --no-capture-output -n polyolefin_ml python -u -m scripts.experiments.run_hierarchy_lipo --action train --variants base_corr full_l1 adaptive --seeds 0 1 --batch-size 32 --micro-batch-size 32
```

`Base+Corr` 只用冻结 Base 的全局表示学习残差，用来控制新增参数容量；`Full-L1` 使用全部一级区域；`Adaptive` 使用自适应 L1/L2/L3 远程上下文。正式编排全程只用 validation 选模，不读取 test 指标。

需要启动或恢复训练时，按 [Frozen 实验说明](docs/FROZEN_MECHANISM.md) 选择 seeds。后台脚本已移到 `scripts/launchers/`；前台命令和后台脚本二选一。

演示默认读取 `diffusion/outputs/ogb_clean/encoder.pt`，下游权重随机初始化，演示输出不是性质预测实验结果。规范化使用原始离散原子属性和键类型，通过 igraph/Bliss 完成，扩散向量不参与规范化。使用 `--variant base` 关闭显式区域/粗边特征及区域大小特征，采用 mean readout；演示的默认配置保留增强项。

## 结果放在哪里

| 结果 | 位置 |
|---|---|
| 当前 Frozen 实验 | `../model/results_formal/05_coarse_gnn/frozen_mechanism/` |
| 正式 coarse 实验及断点 | `../model/results_formal/05_coarse_gnn/` |
| 演示输出 | `outputs/coarse_demo/` |
| 方法检查 | `outputs/method_audit/` |
| 性能报告及历史源码快照 | `outputs/performance/` |
| 演示用拓扑缓存 | `outputs/topology_cache/` |

训练输出沿用原位置；本次整理保留既有权重、结果和数值审计记录。

## 详细说明

- [目录、旧新路径与工具命令](docs/DIRECTORY_LAYOUT.md)
- [Frozen 机制验证协议](docs/FROZEN_MECHANISM.md)
- [第一轮核心消融：代码及执行范围](docs/ABLATION_ROUND1.md)
- [Lipo 正式训练协议](docs/LIPO_FORMAL.md)
- [粗粒度模块接口与缓存](coarse_gnn/README.md)
- [V2 互斥区域与关系增量方法](docs/V2_METHOD.md)
- [扩散子项目](diffusion/README.md) · [编码器核验](diffusion/ENCODER_AUDIT.md)
- [GitHub 上传说明](docs/GITHUB_UPLOAD.md)

新环境依赖见 `requirements.txt`，需要 PyTorch、RDKit 和 `igraph==0.11.9`；不要求 PyTorch Geometric。`conda run --no-capture-output` 用于避免本机 Windows 中文路径的输出转码问题。
