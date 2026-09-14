# 基于粗粒度化的扩散 GNN

可运行的第一版模型框架：完整原图扩散编码 → 分散中心与四跳上下文 → 互斥主归属与残余补充 → 区域 GNN → 粗图 GNN → 图级预测。

新增模块从现有扩散编码器的 `encode_nodes()` 获取节点表示；原始编码器及内部消息门控保留。没有额外的重要节点 Gate、回溯或子图筛选模块。

## 目录

```text
coarse_graining/
├── coarse_gnn/
│   ├── config.py             # 粗化规则和网络参数
│   ├── topology.py           # 中心、归属、残余、上下文与粗边
│   ├── layers.py             # 纯 PyTorch 的带边特征 GIN
│   ├── model.py              # 接收任意节点表示的通用粗图预测器
│   ├── diffusion_adapter.py  # 与现有扩散编码器、分子批次连接
│   └── README.md             # 接口、算法细节、限制与训练示例
├── diffusion/                # 原扩散预训练模块与权重
├── tests/test_coarse_gnn.py   # 拓扑、表示、批次、梯度检查
├── run_coarse_demo.py         # 现有权重加载与前向/反向演示
└── outputs/coarse_demo/      # 演示输出，不覆盖预训练权重
```

## 运行

在本项目根目录，使用已有 `polyolefin_ml` 环境。新增模块只依赖 PyTorch，不要求安装 PyTorch Geometric；SMILES 输入另需 RDKit（已有环境提供）。`--no-capture-output` 避免 Windows 下 conda 对中文路径输出的转码问题。

```powershell
conda run --no-capture-output -n polyolefin_ml python run_coarse_demo.py --backward
conda run --no-capture-output -n polyolefin_ml python run_coarse_demo.py --smiles 'CCO' 'CC(=O)O' 'c1ccccc1' '[Na+].[Cl-]'
conda run --no-capture-output -n polyolefin_ml python -m unittest discover -s tests -v
```

默认加载 `diffusion/outputs/ogb_clean/encoder.pt`，在时间步 0 冻结编码器。无 SMILES 参数时使用 1、24、100 节点的合成链图，输出每张图的归属、上下文、粗边和覆盖统计到 `outputs/coarse_demo/report.json`。`--backward` 使用合成标签执行一次优化器更新，仅用于验证梯度链路。加 `--finetune-encoder` 可检查编码器微调；加 `--device cuda` 使用 GPU。

**下游区域网络、粗图网络和预测头尚未在真实性质标签上训练；演示预测值不能作为性质预测结果。** 当前没有启动正式训练，也未选择下游数据集或目标性质。

通用图接口和训练用法见 [粗粒度模块说明](coarse_gnn/README.md)。原扩散训练与编码用法见 [扩散模块说明](diffusion/README.md)，原编码器核验见 [编码器核验报告](diffusion/ENCODER_AUDIT.md)。
