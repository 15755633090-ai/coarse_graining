# HMG-MRL v2：纠正实验协议，正式训练保持暂停

**继续核对更新：补充材料第 5 节明确所有回归任务 alpha=0.15、tau=0.30。v2 现已显式设置回归温度 0.30，旧分类默认 0.15 不再用于新回归模型。见 audits/hmg_reliability_20261026/FOLLOWUP_PROTOCOL_ALIGNMENT_CN.md。**

2026-10-26。按用户要求，已停止旧队列进程，保留 19/21 个完成结果以及所有已保存权重、曲线和预测。中断位置为 seed 2、AD 组第 44 轮。旧任务的 PAUSED_BY_USER.json 与 results/status.json 记录暂停；不得自动恢复旧协议来填满剩余实验。

## 本次修改

1. **原始描述符**：按已有 DeepChem 200 维列名和排序，重新计算 RDKit 原始值（Ipc 沿用 avg=True），不再把固定 CDF 归一化值当成原始物理量。列名来源与 RDKit 版本、代码、缓存均记录哈希。论文未公开逐列清单，因此这是保持既有列语义的明确实现，不冒称已获得作者完整预处理环境。
2. **训练集标准化**：每个划分独立拟合训练集列中位数、均值、总体标准差；验证/测试只应用。常数列 scale=1；训练集整列无有限值则报错。保存统计与拟合行号，并在运行前用原始矩阵和训练行重新计算核对。
3. **官方初始化**：新建完整模型后调用经官方 Git blob 核验的 initialize_weights，再按种子向七组复制相同参数和 buffer。旧 v1 权重不能初始化或恢复 v2；Full 必须在新协议下重训。
4. **明确训练参数**：Adam、lr=1e-3、batch=64、patience=50，采用公开入口的 weight_decay=1e-5。去掉默认梯度裁剪 5；只有显式设置时才裁剪，仍检查非有限梯度。
   **回归对齐参数**：补充材料已明确 alpha=0.15、tau=0.30；在官方模型构建前设置同一个 cfg，官方 InfoNCE 实际读取 0.30，并记录在协议及基线复用检查中。
5. **五份严格 scaffold 划分**：seed 2020–2024，Murcko includeChirality=False，组内分子不拆开，每份实际 3360/420/420、跨集合骨架交集为 0。生成规则和行号保存；这些不是作者未公开的原始划分，不能声称完全复现。
6. **先 Full 后消融**：baseline 阶段只训练新 ASD；fusion 阶段要求新 baseline 完成，核对数据、源码、标准化、参数及新版本 checkpoint 后才可复用该 v2 Full。不会复用约 0.72 的 v1 Full。
7. **可恢复状态**：每轮保存 latest.pt，包括模型、Adam 状态、完成轮次和历史；中断后从上一完整轮恢复。每轮重新设置配对 RNG，未完成轮重放，不声称能从任意 batch 无损恢复。

原模型的编码器、注意力加权、求和残差、分类头没有替换。七组仍是预测融合子集消融，保留原 A/S 辅助对齐，不能称为三个独立信息源或论文组件删除实验。有效训练容量匹配仍未完成。

## 尚未核实，不能用默认值掩盖

- 作者准确的五份 scaffold 行号和外部 scaffold_split 实现。
- 原 Lipo 回归训练是否标准化标签，以及对应的辅助损失相对尺度。
- 回归学习率调度：正文与公开分类入口 Noam 设置不完全对应。
- 正式回归最大轮数上限；公开分类 config.py 的 epochs/max_epochs=100 已确认，但未证明适用于回归。
- 作者单粒度变体对交互、辅助目标及有效参数容量的具体处理。

因此，v2 标记为 paper-informed，不标记为 exact reproduction。正式 baseline/fusion 入口要求显式传入 --max-epochs、--target-scaling、--scheduler；当前 scheduler 仅实现显式 constant 假设，未擅自套用分类 Noam。正式队列本次没有启动。

## 文件与入口

- hmg_protocol_v2.py：原始列定义、训练集标准化、严格骨架划分、Adam、版本隔离。
- raw_hmg_descriptors_v2.py：不加载 Torch 的原始描述符 CPU 子进程，隔离 Windows OpenMP 冲突，不使用 KMP_DUPLICATE_LIB_OK。
- prepare_hmg_v2.py：只预处理，不训练。
- check_hmg_protocol_v2.py：防泄漏、非有限/常数列、旧权重拒绝、优化器状态恢复检查。
- train_hmg_v2.py：smoke / baseline / fusion 三个阶段。
- train_hmg_fusion_ablation.py：当前转到 v2；旧版本保存在历史运行源码快照与 Git 提交 517df61 中。旧 --reference 命令不再指向当前入口。

```powershell
python -m motif.prepare_hmg_v2 --output motif/runs/hmg_v2_inputs
python -m motif.check_hmg_protocol_v2 --prepared motif/runs/hmg_v2_inputs --output motif/runs/hmg_v2_unit
python -m motif.train_hmg_v2 --stage smoke --prepared motif/runs/hmg_v2_inputs --output motif/runs/hmg_v2_smoke
```

正式运行必须在用户解除暂停、补齐明确设置后执行。模板为：

```text
python -m motif.train_hmg_v2 --stage baseline --prepared INPUTS --output NEW_BASELINE --max-epochs N --target-scaling CHOSEN --scheduler CHOSEN
python -m motif.train_hmg_v2 --stage fusion --baseline NEW_BASELINE --prepared INPUTS --output NEW_FUSION --max-epochs N --target-scaling CHOSEN --scheduler CHOSEN
```

默认经作者官方提交 6ad69261eb1af36c8c0b96594d79a7639b969223 的远端 Git 树核对源码，不修改第三方文件。数据、已有原子/键/Motif 缓存以及新描述符输入另行记录哈希。

## 本机已完成的修改验收

完整预处理：motif/runs/hmg_protocol_v2_prepared_fix1_20261026。4,200 行×200 原始描述符；五个划分的训练集非恒定列均值和标准差检查通过。首次同进程计算触发 OpenMP 冲突的目录保留为失败记录，不能作为已准备输入。

单元验收：motif/runs/hmg_protocol_v2_unit_20261026/acceptance.json。改变验证/测试输入不改变训练统计；常数和非有限处理、统计重载、200 维 MolWt 原始值、五份拟合行号、旧权重拒绝、Adam lr=0.001 均通过。保存第 2 轮优化器状态后恢复第 3 轮，与连续训练参数差为 0（合成小模型验证）。

真实分子冒烟采用 64 条训练、16 条验证、3 epoch，七组全部进行梯度和磁盘重载检查；不测试、不产生性能结论。三轮只有三个完整 batch 更新，不代表收敛。最新验收路径以修改记录中的 acceptance.json 为准。

最新七组验收为 motif/runs/hmg_protocol_v2_smoke_verified_20261026/acceptance.json：passed=true、test_forward=false、formal_training_resumed=false；ASD 与同初值官方完整模型的前向逐值一致。该验收记录了实际 Adam、lr=0.001、batch=64 和各组共享初始化、磁盘严格重载。

上段为温度修正前的历史验收。回归温度修正后的最新独立数值检查为 motif/runs/hmg_protocol_v2_regression_temperature_20261026/acceptance.json；新七组验收路径为 motif/runs/hmg_protocol_v2_smoke_tau030_20261026/acceptance.json。旧验收不冒称已经采用 tau=0.30。

正式阶段缺少未核实配置时，入口已验证会在创建任务目录前拒绝执行；没有默认恢复 100 epoch 或旧目标缩放。旧队列保持 paused，验收后 GPU 处于空闲。

原实验可信性核验及边界见 audits/hmg_reliability_20261026/REPORT_CN.md。原结果只能保留为旧本地协议探索记录，不能与 v2 混合计算种子均值。
