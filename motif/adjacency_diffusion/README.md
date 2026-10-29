# 邻接扩散：MotiL diffusion priming 机制对齐路径

新增“高斯加噪＋整组化学边特征遮蔽”任务以减少目标残留信息，使用 `typed_train --variant masked --pair-mask-ratio 0.3`；性质分支继续使用完整干净键特征。33项测试及小流程通过，尚无新性能结论。设计、兼容性和同架构对照要求见 [MASKED_RECOVERY_20261029.md](MASKED_RECOVERY_20261029.md)。

最新结论：完整键特征版本的迁移诊断发现原早停会漏掉后期改善；统一实际60轮后，性质验证仍是 scratch .7967、full .8183、bounded .8420，未证明预训练增益。当前不能按较高恢复率把 t≤300 视作性质预测最佳设置。详见 [迁移诊断](TRANSFER_DIAGNOSIS_20261029.md)。下文的二值机制和早期协议作为历史实现记录保留。

本模块与 `brics_atom_diffusion` 的原子属性去噪、`context_denoising` 的基序隐向量去噪分开。对齐 MotiL 正文 Methods / Diffusion priming 公式 1–5 的核心：**保留 X，对 A 加高斯噪声，由预测噪声一步恢复 A，以干净邻接重建误差训练**。

**这是论文机制对齐的自建实现，不是完整 MotiL 复现。** 没有实现完整 CMPNN、bi-scaled training，也没有匹配论文全部训练预算/划分。具体数值和接口选择未由正文完全规定的部分在下文披露。来源：https://pmc.ncbi.nlm.nih.gov/articles/PMC12748552/ ，DOI 10.1038/s41467-025-66685-w。

## 核心区别

| 项目 | 当前邻接版本 |
|---|---|
| 干净原子条件 | 八类属性、133 维，逐值核对本地 2a8f006 固定特征函数 |
| 受损量 | 分子内完整二元邻接矩阵，包含有键/无键候选对 |
| 默认时间 | 固定终点 T=1000；random 模式是显式扩展 |
| 恢复输出 | 对称原子对噪声 → 一步还原连续邻接 |
| 训练损失 | 每个分子的非对角 Frobenius 平方和，再按 batch 中分子平均 |
| 编码器可见 | X、A_t、图大小掩码；可选干净 BRICS 归属先验 |
| 编码器不可见 | A_0、真实 edge_index、干净键类型、真实跨边界端点/边界原子标记 |
| 下游 | t=0、干净 X/A_0、相同归属设置；原子读出后性质预测，无反向采样 |

目标邻接使用 0/1 连通性，不恢复键阶或立体键类型。保留的八类原子属性包含总度数、氢数、芳香性、形式电荷等，它们可能提供拓扑线索；这是干净 X 条件的一部分，不能称为无条件从纯噪声重建。

## 信息边界与网络

data.py 明确分成 encoder-facing `Condition` 和 `adjacency_target`：前者仅含 X、节点/候选对掩码及可选同片段关系，没有真实邻接字段。预训练时目标只用于前向加噪、损失和诊断；下游干净预测才使用真实邻接。

传播在分子内**全部有效非对角候选对**进行，包括真实无键位置；连续、可为负的 A_t 作为边输入及消息权重，不做基于真实图的剪枝或阈值化。对称噪声从上三角采样后镜像；解码器输出也显式对称。padding/对角不参加传播、恢复或损失。

节点编码器为自建双状态关系 MPNN。无先验基线的两条路径均全图候选对传播。BRICS 增强版的 s 仅在同片段候选对传播，h 在全部候选对传播；门控仅调节跨片段候选对消息。这里“跨片段候选”包含真实无键，**不是把真实跨键端点提前提供给去噪器**。不创建基序节点，不嵌入片段编号。

BRICS 归属仍从 A_0 生成，携带额外结构信息，必须披露。与统一处理对照共享相同归属，才能区分先验与门控的影响。增强版不能称为无额外条件的严格 MotiL 基线。

## 对齐与自行选择的边界

- 日程按正文式 2 的 `beta_min+(t/T)*(beta_max-beta_min)`（t=1..T），累计 alpha_bar。默认 beta_min=1e-4、beta_max=.02 是明确的实现取值，**未声称已经从正文确认这些数值**。配置保存离散方式与 alpha_bar_T。
- 默认 terminal 模式对应正文式 3–5 的终点 T 恢复。`--time-mode random` 使用随机时间步，但仍以 A 重建训练；这是变体，不是均匀 epsilon MSE。
- 计算式 `A_hat=(A_t-sqrt(1-alpha_bar_t)*epsilon_hat)/sqrt(alpha_bar_t)`，不裁剪 A_hat 后再训练。epsilon MSE 仅诊断，不参与选模或总损失。
- 使用二元无向邻接、上三角镜像噪声、对角/padding 掩码、每分子完整非对角 Frobenius 损失和 batch 均值，是公开的实现选择；同一无向对在损失中计两次，而分类指标计一次。
- 时间嵌入、自建消息算子/对解码头、层宽、batch size、梯度裁剪及后续目标标准化，均不等同于已核验的完整官方实现。预训练使用 Adam、默认 lr=3e-5；不将小预算说成论文 250K 规模复现。

## 三组配置及对照

| 配置 | 开关 |
|---|---|
| 无 BRICS 的邻接机制基线 | 默认，不加开关 |
| BRICS＋统一跨片段处理 | --motif-prior |
| BRICS＋条件门控 | --motif-prior --conditional-gate |

`--conditional-gate` 必须搭配先验，否则拒绝。可选 `--single-state` 为另一结构控制。移除门控会改变参数量和随机初始化消耗；不是容量匹配对照。预训练和微调的配置必须一致，不允许把属性去噪或旧基序模型的检查点迁移过来，即便输入维度碰巧都是 133。

```powershell
$py = 'C:/Users/12775/miniconda3/envs/polyolefin_ml/python.exe'
& $py -m unittest motif.adjacency_diffusion.test_interfaces -v

# 在项目根目录运行。语料/划分/预算先冻结，默认只用无标签验证集选模。
& $py -m motif.adjacency_diffusion.pretrain --smiles-csv path/to/train.csv --valid-csv path/to/valid.csv --output-dir motif/adjacency_diffusion/runs/baseline_pretrain --device cuda:0

# 同架构：预训练后微调 vs 从头训练；默认没有测试评价。
& $py -m motif.adjacency_diffusion.finetune --dataset esol --pretrained motif/adjacency_diffusion/runs/baseline_pretrain/best.pt --learning-rate .001 --epochs 100 --output-dir motif/adjacency_diffusion/runs/esol_pretrained --device cuda:0
& $py -m motif.adjacency_diffusion.finetune --dataset esol --learning-rate .001 --epochs 100 --output-dir motif/adjacency_diffusion/runs/esol_scratch --device cuda:0

# 门控增强组必须在两阶段都加这两个参数，并使用它自己的预训练权重。
# --motif-prior --conditional-gate
```

语料读入、规范去重、预训练 train/valid 重叠拒绝、下游 valid/test 语料暴露检查、官方任务读入/评价和 IO 事务复用已验收的通用工具。新模型/日程/目标/指标/检查点 schema 独立实现。`--resume` 要求原命令和源代码/配置不变，恢复以包含最佳快照的 latest.pt 为唯一依据，修复 best/history 导出。完整数据和权重不随源码必然提供。

## 指标与资源

选模指标为每分子 Frobenius 重建误差。另记无向候选对 MSE、噪声 MSE、正边召回/精度/F1、特异度、balanced accuracy，以及汇总候选对的 ROC-AUC/AP。阈值固定 .5，不用测试集挑选。给出全零邻接的 Frobenius 基线，避免仅靠多数无键得到高准确率。按相对时间区间分组；terminal 模式下较低区间为空是预期情况，不填零。

全候选对编码有 O(B*N²*d) 的内存开销，batch size 应按图大小规划，不能以真实边稀疏支持偷换计算来省显存。先在固定小语料验收，再冻结正式预算。

## 带键类型和完整键特征的实现

当前推荐使用 `typed_train` 入口，保留原子特征，并在去噪与干净预测中显式使用键类型、立体、共轭和环信息。二值邻接实现作为历史对照保留。详细格式、命令与验收见 [TYPED_BONDS_20261029.md](TYPED_BONDS_20261029.md)。新旧检查点和恢复损失不兼容，旧结果不能代替新版本验证。

带键特征的充分训练及三个修订对照已完成8组，分层/证据改动未全面胜出；具体结果、稀有类别局限及限定噪声范围的补充实验见 [TYPED_RECOVERY_REVISION_20261029.md](TYPED_RECOVERY_REVISION_20261029.md)。

限定训练噪声范围的两组补充对照也已完成：同一 t200 条件下，类型/联合正确率约90.8%/90.2%，但训练范围外高噪声表现更差。后续优先验证 `--variant frequency_init --training-max-t 300`，并保留全范围参考；尚无该设置的下游性能结论。

已固定同架构 scratch/full/bounded 的三种子 ESOL 验证协议，见 [TYPED_PROPERTY_PROTOCOL_20261029.md](TYPED_PROPERTY_PROTOCOL_20261029.md)。运行中不修改噪声范围、读出或门控，不用部分结果提前判断新版下游收益。

该九组验证已全部完成：验证 RMSE 均值 scratch .8166、full .8626、bounded .8561；未显示新版预训练的下游收益，bounded 三个种子均未优于 scratch。完整配对差值、立体属性披露和适用范围见上述协议与运行结果。

后续诊断已补齐冻结探针、18组冻结预测头、6组编码器LR消融和7例续训：全部训练满60轮后均值为 scratch .7967、full .8183、bounded .8420。原早停掩盖部分晚期改善，且full/bounded排序反转；限定噪声的恢复优势仍未转化为性质收益。证据和未证实机制见 [TRANSFER_DIAGNOSIS_20261029.md](TRANSFER_DIAGNOSIS_20261029.md)。

## 已完成验收（不是性能结论）

新增有限预算恢复筛选已完成，后续配对性质验证独立记录，见 [RESCUE_SCREEN_20261029.md](RESCUE_SCREEN_20261029.md)。实验适配保留主实现默认配置；不能把有界参数化、额外 BRICS 条件及门控的联合收益解释为单独门控贡献。

第二轮噪声证据跳接及受损输入对照 8/8 已完成；综合指标未支持替换当前方案，完整失败结论和对照见 [POSTERIOR_REVISION_20261029.md](POSTERIOR_REVISION_20261029.md)。

补充定位：固定 8 分子、三种噪声强度、两种参数化、两档学习率的 12 格诊断已完成，主实现未改。结论及不能外推的边界见 [TERMINAL_DIAGNOSTICS_20261029.md](TERMINAL_DIAGNOSTICS_20261029.md)。

当前本地 11 项测试全部通过：固定官方八类属性逐值核对；真实非键参与加噪；噪声/输出对称；固定 A_t 下真实边/干净键不形成旁路；A_t/time 梯度与响应；已知噪声恢复及正确损失权重；损失确为 A 重建；原子重排/编号重标；内部隔离、全候选门控；合批/padding/单原子/干净预测不采样；全零基线的召回诊断；两阶段门控梯度与统一门控。（部分检查合并在同一测试中。）

官方特征逐值测试依赖本地固定 2a8f006 源文件；不包含该外部快照的克隆会跳过这一个核对，其余检查可独立执行。

`runs/smoke_20261029` 使用已排除 ESOL 全部分子的固定 ZINC 子语料，仅取 train/valid 各 16 分子、CPU、seed7、hidden12/layers2，三种预训练配置各两轮；另有 ESOL 基线迁移/从头各两轮、门控迁移一轮，以及 BBBP 从头一轮。全都只看验证集。

acceptance.json 记录：latest 提交后/best 未导出时模拟中断，并置入 epoch999 的错误 best/history；恢复权重、历史和 RNG 与连续预训练一致，导出恢复正确。四份下游最佳检查点重载复现验证指标，属性去噪旧检查点被拒绝。

终点 T=1000 的两轮小训练重建误差仍远高于全零邻接基线。**这只说明代码路径和诊断可用，尚不能声称模型已经学会恢复连接、门控有效或达到论文成绩。** 未启动正式论文对标训练，也没有运行完整 MotiL 三阶段基线。
