# V2：互斥区域与关系增量粗图

V2 是新的实验身份；历史 V1 粗化、结果与恢复协议保持不变。

## 固定主方法

1. 使用现有 4 层、128 维、带键信息的 diffusion encoder 生成原子表示。
2. 令 `s=8`，初始中心数为 `max(C, ceil(N / s))`。每个连通分量先获得一个中心，剩余预算按分量节点数使用最大余数法分配。
3. 每个分量首中心取图中心，其余中心采用 farthest-first。
4. 使用确定性多源逐层扩张形成完整、互斥、连通的主区域。
5. 区域回正使用诱导子图距离。固定中心数时最多两轮，只有目标
   `(最大半径, 半径和, 无权切边数, 区域大小平方和, 规范中心元组)`
   严格改善才接受。
6. 区域诱导子图半径上限为 4。违规时一次只在最严重区域的最远点补一个中心，立即重新全图划分和回正。
7. 默认无额外上下文：`context == core`。不存在 residual 区域。
8. Region 使用两层带真实键类型和残差连接的 GINE，随后对区域成员取 mean 得到 `z0`。
9. Region 图表示使用 core size 加权平均；size 默认不作为节点输入特征。
10. 跨区域物理原子键形成粗边，粗边编码为
    `[log(1+count), single_ratio, double_ratio, triple_ratio, aromatic_ratio]`。
11. Full 模型使用一层接收端、发送端与粗边联合消息网络，按粗邻居数量求平均，仅生成关系增量 `delta_z`。无邻居节点的增量严格为零。
12. 最终预测为
    `region_head(g_region) + alpha * (delta_head(g_delta) - delta_head(0))`，
    其中 `alpha` 可训练且初始化为 0.1。

## 核心模型

- `v2_region_only`：相同 Region 主路径，关闭 coarse relation layer。
- `v2_full`：完整 V2 Region 主路径与 Coarse 关系修正。
- `Coarse-no-edge` 不是独立训练模型；按照硬约束它必须与 Region-only 等价，应作为测试检查。

## 当前入口

```powershell
# 仅构建缓存并写拓扑 sanity 汇总，不训练
python run_lipo_v2.py --action prepare --seeds 0

# 正式训练入口；按需要指定 1/3/5 个 seeds
python run_lipo_v2.py --action train --seeds 0
```

当前 `train` 入口明确使用 `pretrained_finetune`：encoder、Region、Coarse 和 heads
从第一步开始联合优化。它用于与已有端到端 baseline 做匹配比较，不代表“先冻结、
后解冻”的两阶段上限训练。

## 后续消融

- 结构：0/1-hop context，Coarse 0/1/2 层。
- 性能增强：core size feature、`log(1+N)`、Coarse sum/mean。
- 上限训练：`s`、最大半径、hidden width、学习率、dropout、weight decay，以及分阶段 encoder fine-tune。

本轮代码只实现冻结默认 V2、Region-only 对照、Full 模型和拓扑准备入口；不改变历史 V1 实验定义。
