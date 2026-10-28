# 正式训练前的控制与追溯修订

针对 a94c35e 的静态审查，保留 squared-L2 去噪目标、lambda 和联合全局梯度裁剪，新增诊断与文件保护。完整原子 GNN 已传播目标信息、单基序样本没有基序间上下文这两项方法限制仍存在。

## 修改

1. 每步记录裁剪前总范数、裁剪系数、epoch 裁剪触发比例。默认每 10 个优化器步以 autograd.grad 抽样记录性质／辅助原始及加权梯度范数、共同到达参数上的梯度比。第一步必采；诊断不消费 RNG 或写入 .grad。
2. 单／多基序目标分别记录误差总和与目标数；另记按分子等权的诊断误差。跨 batch 用总和／总数聚合，训练仍按所选目标平均。
3. 同一输出目录的协议固定数据、划分、源码、软件版本及训练设置。追加新种子／配置时扫描全部已完成运行重建汇总；原 run 禁止覆盖；协议不一致、历史无协议目录和分解审计变化均拒绝混合。独占会话锁与临时文件替换避免并行覆盖或半写入。
4. 源码哈希覆盖 benchmark.py、基线模块和 bond_diffusion；记录 Python／RDKit／PyG／OGB 等版本。Git SHA 与精确源码的匹配情况单独标记，避免将旁边 checkout 的提交冒称运行源码版本。参数量分别记录实例化总量、当前优化路径和实际收到梯度的数量，无关模块冻结并排除出优化器。

## 验收

- `python -m unittest motif.context_denoising.test_method motif.context_denoising.test_controls -v`：14 项通过。新增测试验证诊断前后 RNG、最终梯度和一次 AdamW 更新逐值一致，分组损失分母、参数路径、依赖追溯、追加汇总、防覆盖与会话锁。
- `runs/controls_fix_20261028`：先 context，后 fixed/random/no_communication，两次真实命令调用，ESOL 每 split 前 32 个样本，seed 0，各一轮，仅验证集。汇总保留全部 4 次完成记录。`controls_acceptance.json` 记录重复 run 与协议改变均被拒绝，且拒绝前后已有文件 SHA-256 不变。
- `runs/controls_classification_20261028`：BBBP random，每 split 前 128 个样本，一轮，仅验证集；分类梯度诊断和保存流程通过。
- 这批验收设置 `--gradient-diagnostics-every 1`。ESOL 每个 run 两个优化器步均采样；规模太小，不用于估计正式训练中的裁剪比例或确定 lambda。

早期 `SYNC_MANIFEST_20261028.json` 仍保留为 a94c35e 的历史快照清单；修订快照使用新的清单文件，不重写历史结果。此次没有启动正式五种子训练，没有读取新的测试结果，也没有建立上下文／扩散预测增益。
