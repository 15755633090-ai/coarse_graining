# 固定拓扑全图属性扩散：首版实现

本入口与旧的基序类型条件全图扩散分开，checkpoint schema 不兼容。不改动旧结果或原论文源码。

## 方法边界

- 输入：全分子二维图；元素、形式电荷，真实边的键类型（单/双/三/芳香/其他）。连接关系固定，只沿真实边传播。无连接恢复目标，不预测无键类别。
- 原子属性全部联动遮蔽；被遮蔽原子的所有相邻键一起遮蔽。每个物理键只存一份，传播时复制为两个方向。省略芳香性、杂化、氢数、价态等冗余原子特征。固定拓扑仍会提供度数和环连接信息，这是条件的一部分。
- t 均匀采自 1…T；累计遮蔽率 q(t)=q_max(1−cos(πt/T))/2。掩码为吸收态，可用共享均匀阈值构造随 t 嵌套的前向过程。区域部分：以固定概率选择一个随机种子 BFS 邻域，再以 q(t) 激活整个邻域遮蔽；不使用基序标签或性质标签。默认随机遮蔽为主，区域概率 0.2、区域最大目标大小为分子的 20%；断开图中的 BFS 不跨连通分量。
- 预测原始属性 x0。损失：各原子属性在受损原子上分别平均，再对属性平均；键损失在受损物理键上平均；两者加权后对分子平均。没有受损位置的项为零，不强制补采掩码。**这是多时间步吸收态去噪的 CE 替代目标，不是精确 ELBO，也不把它称为已实现的生成采样模型。**
- 下游干净输入使用最低已训练时间 t=1。存在干净输入与低噪声输入的分布差异；没有宣称零噪声条件得到专门监督。
- 预训练完全不读取两套基序划分；按原数据 train 学习，valid 固定逐行噪声选模。仅使用全部划分的分子身份检查跨划分重叠，重叠即停止；不使用 test 属性或性质标签优化/选模。当前身份检查为规范化立体 SMILES，不等于盐、互变异构体或 scaffold 去重。
- 下游保留两篇论文的成员集合和 token/node 顺序，逐成员均值读出。**成员均值是本版本新增读出，不宣称完全复刻原论文的基序特征构造。** HMG 在 frag_input_projection 输出处加投影残差；HimNet 在原基序 f_atoms 行加投影残差。没有新增全分子池化预测分支。
- 融合保留原向量，逐通道 sigmoid 门初始为 0.1；联合微调。默认编码器学习率为其余网络的 0.1 倍。新任务禁止额外辅助损失，三个组都将原论文已有辅助损失权重置零，因此 baseline 是此统一协议下的原架构，不能冒充论文完整训练协议复现。
- HMG 原有构象/描述符分支保留；新增分支只用二维信息，不能称整个 HMG 模型完全二维。

## 运行

工作区根目录，Python 使用已有 polyolefin_ml 环境：

```powershell
$motifPython = 'C:/Users/12775/miniconda3/envs/polyolefin_ml/python.exe'
& $motifPython -m motif.check_fixed
& $motifPython -m motif.check_paper_adapters --paper him --fixed-topology
& $motifPython -m motif.check_paper_adapters --paper hmg --fixed-topology
& $motifPython -m motif.pretrain_fixed --data motif/data/lipo.pt --output motif/runs/fixed_pretrain_seed0 --epochs 50 --seed 0
& $motifPython -m motif.launch_fixed --paper HimNet --paper-root coarse_graining1/HimNet_reproduction_20261013/windows --data motif/data/lipo.pt --checkpoint motif/runs/fixed_pretrain_seed0/best.pt --output motif/runs/fixed_him_3arms --seeds 0 1 2
& $motifPython -m motif.launch_fixed --paper HMG-MRL --paper-root research_code_audit/HMG-MRL --hmg-feature-cache motif/runs/hmg_features/features.pickle --data motif/data/lipo.pt --checkpoint motif/runs/fixed_pretrain_seed0/best.pt --output motif/runs/fixed_hmg_3arms --seeds 0 1 2
```

HMG 的缓存必须使用已有官方预处理缓存的实际路径；示例路径不保证存在。原模型需要其既有依赖，HMG 需要 CUDA。三组 original/random/pretrained 使用相同 backbone 初始状态并逐张量核对；random 只读取 checkpoint 的结构与数据来源，不加载其权重。默认复用一个预训练种子，不覆盖预训练种子间不确定性。

pretrain 的 best.pt 由 valid 损失选择，last.pt 保存优化器与 RNG，但尚无恢复训练命令；不能声称已验证完整续训。下游保存训练集标签归一化参数、源码/数据哈希、划分行号、权重和选模后一次测试预测。当前下游仅支持既有 Lipo 单任务回归。

## 验证范围（2026-10-18）

新模块检查已通过：不可变拓扑、相关属性遮蔽、边方向对称性、原子置换等变性、重叠成员读出、非基序位置不变、性质损失到编码器梯度、单原子及无边图。

两篇真实模型类均完成合成输入前后向检查。另完成 32 train + 32 valid 分子的 1 epoch 预训练，以及每篇模型 2 train + 2 valid + 2 test 真实分子的 1 epoch 预训练分支训练、选模、重载、测试；结果目录分别为 fixed_topology_smoke_20261018、fixed_him_pretrained_smoke_20261018、fixed_hmg_pretrained_smoke_20261018。全部为 smoke，不能用于性能收益结论。正式三组多种子训练尚未开展。

三组启动器也已分别完成两篇模型的 seed=0、每划分 2 分子、1 epoch 检查，目录为 fixed_him_three_arm_smoke_20261018 与 fixed_hmg_three_arm_smoke_20261018。每组都完成验证选模、最佳权重重载与最终测试，completed.json 记录三组原骨干初值逐张量匹配。随机组确认不加载预训练权重。这些测试验证可执行性和对照配对，不验证性能收益。
