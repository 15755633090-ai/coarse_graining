# 规范化修复验收（2026-09-14）

本次只处理带原子/键属性图规范化，以及主模型增强项解耦。残余补充策略保留，未开展真实性质训练。

## 方法

原始五类离散原子属性作为原子颜色；每条物理键转换为具有键类型颜色的辅助节点。原子和键的颜色空间分离。使用 igraph 0.11.9 的 Bliss（`sh="fl"`）得到规范原子顺序后，回到原始分子键拓扑运行原粗化算法。辅助节点不参与四跳距离和 GNN。

规范化不使用扩散向量、浮点舍入或 persistent ID。输出保留输入编号与规范编号之间的双向映射。

## 审计结果

每种配置、每张图随机重排 100 次，固定下游初始化种子 42，调用 `eval()`。修复后每次重排均重跑扩散编码器、规范化和全部粗化。验收要求结构完全相同，预测绝对变化不超过 `1e-6`。

| 输入 | 修复前粗节点数范围 | 修复后粗节点/粗边数 | CPU 增强版最大预测变化 |
|---|---|---|---|
| 24 节点链 | 4～5 | 5 / 4 | 0 |
| 100 节点链 | 13～20 | 17 / 16 | 0 |
| 分子图示例 | 3～5 | 4 / 3 | 1.12e-7 |

这些输入仅用于重编号不变性审计，不代表下游数据集的选型。项目面向普通分子数据集；样例名称调整不改变输入图及已有审计数值。

三组输入的规范粗化签名均完全一致。签名包括原始离散属性、归属、中心、上下文、残余标志、粗边和跨区边数，验收不是只比较节点/边数量。

- CPU 增强版：[canonical.json](canonical.json)，最大绝对变化 1.12e-7。
- CPU 基础版：[canonical_base.json](canonical_base.json)，最大绝对变化 8.94e-8。
- CUDA 增强版：[canonical_cuda.json](canonical_cuda.json)，最大绝对变化 9.69e-8。
- 修复前报告：[current.json](current.json)。修复前编码器只单独核验了每组前三次重排，主审计重排缓存节点表示；修复后使用更严格的全流程重算。

16 项单元/方法测试通过，其中不变性测试对六种图各重排 100 次；增强开关覆盖 32 种组合。原始属性和键类型保留、无向边双向存储、映射回输入、节点/边梯度也经过测试。另使用现有预训练权重完成 CUDA 基础版编码器微调和一次优化器更新，见 [运行报告](../coarse_demo/canonical_base_finetune_cuda.json)。

## 适用范围

以上证明实现通过这些输入上的不变性验收，不代表下游性质精度或长程性能已经提升。数学上保证的对象是完整带属性图的规范形式和等变编码器产生的图级预测；对称原子映射回原编号时可互换，不要求固定某个外部 atom ID 当选中心。浮点并行计算仍允许数值误差。

Bliss 最坏复杂度为指数级。以上审计产生时尚未增加规范化缓存；后续实现的缓存与预计算接口见 [模块说明](../../coarse_gnn/README.md#拓扑缓存与预计算)。

## 复现

在项目根目录运行：

```powershell
conda run --no-capture-output -n polyolefin_ml python -m unittest discover -s tests -v
conda run --no-capture-output -n polyolefin_ml python -m scripts.validation.audit_method --permutations 100
conda run --no-capture-output -n polyolefin_ml python -m scripts.validation.audit_method --permutations 100 --variant base --output outputs/method_audit/canonical_base.json
conda run --no-capture-output -n polyolefin_ml python -m scripts.validation.audit_method --permutations 100 --device cuda --output outputs/method_audit/canonical_cuda.json
```
