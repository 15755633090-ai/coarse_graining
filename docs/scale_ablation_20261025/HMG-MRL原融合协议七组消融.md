# HMG-MRL 原融合协议七组消融

日期：2026-10-25。用户已选择沿用原模型训练和融合协议，进行七组尺度消融，而非冻结编码器后换新拼接预测头。

## 与 HimNet 冻结探针的区别

此前 HimNet 完整模型 seed 0 的保存测试预测重算 RMSE 为 0.714437；本轮冻结探针 ASG 为 0.9461±0.0163，二者并非相同计算图。冻结探针去掉指纹预测支路、改为各尺度独立平均与映射、拼接新预测头，不能把 0.9461 当成原完整模型复现结果，也不能直接归因于某一个改动。原报告现已补上原完整模型参考和可比性说明。

## HMG 分支定义

- A：原 `mol_feature_by_atom`。
- S：原 `frag_cls`，包含作者原 Motif 输入、对原子的交叉注意力及 Motif Transformer。
- D：原 `mol_feature_by_des`，200 维描述符经原 MLP 和原子上下文更新；不是 HimNet 全图节点，不能混称 G。

七组为 A、S、D、AS、AD、SD、ASD。保留完整原输入和分支计算，只将进入原融合层的视图改为子集。融合仍按所选视图重新求和、重新归一化注意力，再经过原激活和原分类头；不放零占位视图，不引入新拼接 MLP。ASD 使用原 Full forward，不改计算图。

所有组继续使用原 A/S 辅助对齐，避免同时改变训练目标。因此是“参与预测融合的尺度消融”，不是删除结构分支或消除全部对应尺度信息。D 也可能通过对原子的上下文查询包含原子信息。

## 已锁定协议

沿用此前 HMG 原架构的统一监督回归协议，不称原论文全部预训练流程复现。原 Lipo 划分 3360/420/420，seed=0/1/2。每个种子七组读取同一份已保存原始初始化，所有参数和 buffer 逐 tensor 核对相同。

保持原半径 2、原预编码器和 Motif Transformer 层数、原 dropout=0.3、原融合层和原预测头。AdamW lr=1e-4、weight_decay=1e-5、batch=16、梯度裁剪 5、原辅助系数 0.15、100 epoch 上限、patience=20。每个配对种子使用相同批行号与 epoch RNG 重置规则。标签标准化仅用训练集；验证集严格选模、磁盘重载后最后测试一次。

保留作者原构象相关键输入、描述符和已有训练集统计填补后的有限值缓存；核对数据、缓存、源码及初始权重哈希。不新增扩散模块，不改分组，不根据测试成绩延长预算。

七组注册参数数目和融合/预测头形状不变，但未进入损失路径的参数不构成有效容量。每组保存 gradient_audit.json，区分有梯度参数和注册参数；本轮不宣称已做有效容量匹配。

## 原 Full 结果复用与新训练

来源为 motif/runs/scale_pairs_3seeds_20261018 的三个 hmg_seed 运行。先核对原模型源码、划分、特征、原始初始化和超参数，再重载原 Full 的最佳权重检查验证 RMSE 一致。原测试预测只重算保存指标，不再执行测试 forward。满足这些条件才将原 Full 记为 ASD。

三份原 Full 基线复用，其余六组×三个种子共 18 次新联合训练，形成 21 个配对结果。每个种子先验收 ASD 基线，再执行该种子的新增组；失配或非有限值会停止队列。

## 验收与执行状态

七组均通过 32 train / 16 valid、3 epoch 的真实分子冒烟，未测试。核对 ASD 与原 Full 的前向逐值相同、各组辅助损失不变、梯度有限、磁盘保存后扰动参数并严格重载能恢复验证预测。记录位于 motif/runs/hmg_fusion_7arms_smoke_v2_20261025/acceptance.json。

正式任务已通过隐藏后台进程启动，任务目录 motif/runs/hmg_fusion_7arms_3seeds_20261025。执行前复制顶层 Python 源码到 source/motif，训练从该快照运行，后续修改工作文件不会改变当前任务。

- launch.json：启动参数与 PID。
- worker.stdout.log / worker.stderr.log：训练日志和异常。
- results/status.json：实时阶段、seed、arm、epoch、batch、完成数；complete 才代表全部完成。
- results/protocol.json：锁定协议和哈希。
- results/seed_N/组别/：forward.py、初始化/梯度记录、曲线、最佳权重、预测。
- results/completion.json：全部运行完成标记。
- results/summary.csv、paired_deltas.csv、REPORT_CN.md、artifact_audit.json：队列完成后自动生成并核对。

入口支持 --resume：核对协议和哈希后跳过完成的组，未完成组从同一原始初始化重新训练，不把只含最佳权重的文件冒充精确优化器续训。

启动核对：首个沙箱后台进程未进入训练，已改为获批的主机隐藏后台进程。launch.json 当前 PID=34964。原 ASD seed 0 重载验证 RMSE=0.70542984，与历史一致；保存测试预测重算 RMSE=0.72122422。新增 A seed 0 已完成第 1、2 轮并进入第 3 轮，stderr 为空；此记录只证明任务实际运行，不代表全部完成。完整队列完成前以 results/status.json 为准。
