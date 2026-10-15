# bca8651 审查后的修订

1. **损失口径修正。** 每个 Motif 内分别计算原子平均、键/无键分层平均，然后等权平均 Motif；单原子边项为零。训练日志、验证选模均按 Motif 数正确累加。验证噪声额外按样本身份固定，使改变 batch 边界也不会改变噪声样本。
2. **共享分桶与分块。** 预训练采用大小分桶；两套适配器、独立语境编码、导出共用 encode_motifs，保持输出成员顺序。支持 padded 原子对预算；超预算的单 Motif 报错而不丢弃。有梯度编码使用 activation checkpoint，减少跨分块保留的激活。
3. **真实论文训练入口。** 新增 train_paper.py，调用实际 HMG-MRL/HimNet 类及其特征路径；不使用独立 context.py 冒充论文模型。支持 original/pretrained 两种入口、原划分、train 标准化、验证选模与一次最终测试。
4. **官方缓存生成。** 官方 README 表示特征字典须外部准备，未随仓库提供。新增 prepare_hmg.py 调用官方 save_smiles_dicts；保留构象、MMFF 和归一化描述符，不以新二维特征代替原模型输入。使用官方路径生成了 6 个真实分子的检查缓存，所有请求行号与 200 维描述符验证通过。

## 已运行的检查

- 原实现检查：通过。
- 分批前后损失与验证噪声一致：通过。
- 分桶覆盖、原子对预算与超预算失败：通过。
- 分块编码的输出、顺序及梯度一致：通过。
- 真实分子新旧 HimNet BatchMolGraph 输入逐位一致：通过。
- 两套真实论文模型分别在 6 个 Lipo 分子（train/valid/test 各 2）上训练 1 epoch、重载、最终测试：通过，均标记 smoke。
- 官方源码：16 个既有审计文件哈希保持不变。

没有开展正式性能比较，没有建立增益结论。原模型运行环境与原论文统计复现是不同事项；本次使用原工作区已有 Windows 兼容版 HimNet 和固定官方 HMG-MRL 模型，记录实际源码哈希。

运行说明见 PAPER_TRAINING.md；本地检查证据在 runs/revision_checks.json、runs/paper_data_check.json、runs/him_real_smoke/、runs/hmg_original_features_smoke/ 和 runs/hmg_real_smoke/。这些模型及数据产物均不上传 Git。
