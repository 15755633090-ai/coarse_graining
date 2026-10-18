# 固定三种子残差融合比较

仅修改接入机制，不改均值池化、类型词表、扩散算法或预训练目标。类型/全图编码器继续联合微调。

HMG-MRL 在原 frag_input_projection 输出处加残差；HimNet 在原基序 f_atoms 行加残差。公式为 original + sigmoid(a) * projection(diffusion)。a 是逐通道可学习参数，在所有基序间共享，初始 sigmoid(a)=0.1；投影为普通非零随机初始化。该版本尚未使用依赖具体基序内容的动态门控。

训练入口：`motif.train_paper --fullgraph --fusion residual --gate-init 0.1`，其余参数与旧正式比较一致。默认 `--fusion replace` 兼容既有权重。残差权重需要以相同 fusion 配置重建模型再 strict 加载。

固定实验目录：`motif/runs/fullgraph_residual_3seeds_20261016`。包含两个真实模型检查和14个新正式下游运行。每篇模型比较 original/replace/residual，seed=0/1/2；seed0 原版和替换复用已完成结果，新增其残差组；seed1/2 各训练三组。每个种子逐项检查 backbone 初始状态完全一致。

所有种子复用各自论文方法已经完成的 seed0 预训练 best.pt。因此只衡量下游初始化/训练随机性，不涵盖预训练随机性。既有测试结果已被查看，本轮不是全新盲验证。固定 gate=0.1、100epoch、patience20、lr1e-4、batch16；不按测试分数调整设计、种子或预算。各组测试仅在验证选模结束后执行。

查看 status.json、completed_results.json、comparison.json。发生任何子进程失败或 backbone 初始化不配对时队列停止并保存错误。source/ 是冻结的完整运行源码，根目录后续修改不影响本次训练。
