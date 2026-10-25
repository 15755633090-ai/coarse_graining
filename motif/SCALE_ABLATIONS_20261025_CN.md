# 三尺度实验代码与协议边界

本次同步只包含源码和说明，不包含数据集、模型权重、特征缓存、历史运行目录或完整第三方仓库。实验输入仍需从已核验的本地归档提供。

## 两种实验不得混称

| 实验 | 表示来源 | 预测方式 | 训练范围 |
|---|---|---|---|
| HimNet 冻结读出探针 | 已训练的候选修正版 HimNet 双路径节点状态 | A/S/G 分别汇聚、共享映射后拼接新 MLP，不接指纹预测支路 | 只训练预测头 |
| HMG-MRL 七组融合消融 | 原模型原子 A、Motif S、描述符 D 分支 | 原注意力融合、原预测头；只切换进入融合的视图子集 | 从同种子原始初始化联合训练 |

HimNet 原完整模型 seed 0 保存测试预测重算 RMSE 为 0.714437；冻结探针 ASG 为 0.9461±0.0163。这不是同一模型复现失败的直接证据：冻结探针改变了指纹支路、读出、预测头及训练范围。不能把差异单独归因于其中一个改动。完整的可比性说明见 docs/scale_ablation_20261025/第一轮三尺度预测对比.md。

## HimNet：冻结读出诊断

- `him_scale_readout.py`：不修改上游文件，暴露双路径平均前节点状态；按原节点类型读出 A/S/G。
- `check_him_scale_readout.py`：合成扩展图上的原向量恢复、冻结和梯度检查。
- `smoke_him_scale_probes.py`：真实分子小样本流程验收，不进行测试集 forward。
- `train_him_scale_probes.py`：锁定编码器与 Lipo 划分，七个主组、六个去重容量对照、五个预测头种子。
- `report_him_scale_probes.py`：从保存预测核验 RMSE 并生成报告；直接以脚本执行，避免包初始化额外加载 Torch 后产生绘图 OpenMP 冲突。

这些入口依赖归档的修正版源码、Full checkpoint、图缓存和划分 manifest；不是下载仓库即可从零运行的独立数据管道。默认路径反映本地实验布局，可用入口参数指定相应运行目录；归档 manifest 内绝对输入路径也须在迁移机器时核对。

## HMG-MRL：沿用原融合的七组消融

- `hmg_fusion_ablation.py`：生成 A、S、D、AS、AD、SD、ASD 七个 forward；仅改变原视图 stack，随后原求和残差、注意力与分类头保持不变。ASD 使用原 Full forward。
- `train_hmg_fusion_ablation.py`：原架构在统一监督协议下的端到端训练，原 A/S 辅助对齐目标对所有组保留。
- `report_hmg_fusion_ablation.py`：全部结束后自动核验保存预测、汇总配对结果，不重新执行测试。

D 是 200 维描述符支路，不是 HimNet 的全图节点。各组仍计算原分支并保留辅助对齐，因此是预测融合消融，不是完全删除某类结构或信息。注册参数数目相同不等于有效容量已经匹配，实际梯度路径会另外保存。

沿用本地此前的统一 Lipo 监督协议：seed 0/1/2、batch 16、AdamW lr=1e-4、weight_decay=1e-5、梯度裁剪 5、100 epoch 上限、patience 20、原辅助系数 0.15。该协议尚未逐项证实等同论文全部实验设置，不称论文完整训练流程复现。

先运行无测试的冒烟验收，再启动正式训练：

```bash
python -m motif.train_hmg_fusion_ablation --reference /path/to/verified_reference_run --smoke --output /path/to/smoke
python -m motif.train_hmg_fusion_ablation --reference /path/to/verified_reference_run --smoke-result /path/to/smoke --output /path/to/formal
```

`verified_reference_run` 必须包含三个 `hmg_seedN` 的原始初始化、完整模型最佳权重、验证/测试保存预测与 manifest。manifest 记录的原论文源码、Lipo 缓存和有限值特征缓存必须可访问，且哈希与行号一致。模型使用原 CUDA mask 工具，当前入口要求 CUDA；不静默改写 CPU 实现。

原 ASD 基线仅在源码、输入、划分、初始化和协议一致、重载验证一致后复用，其他六组×三个种子新增训练。`--resume` 跳过已完成组；中断组从同一原始初始化重训，不冒充优化器精确续训。

正式任务的 `status.json` 标明当前阶段；只有 `completion.json` 为完成记录。训练结束自动生成 `runs.csv`、`summary.csv`、`paired_deltas.csv`、`REPORT_CN.md` 和 `artifact_audit.json`。运行中日志不作为最终多种子结论。

进一步的实验说明见 docs/scale_ablation_20261025/HMG-MRL原融合协议七组消融.md。文档中的本机运行路径和阶段性状态仅用于追溯，不是 GitHub 实时训练状态。
