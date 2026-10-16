# 带基序结构类型条件的全图扩散

新入口为 `python -m motif.pretrain_fullgraph`。旧 `pretrain` 仍表示孤立基序高斯扩散，不能混用其权重。

```powershell
$py = 'C:/Users/12775/miniconda3/envs/polyolefin_ml/python.exe'
& $py -m motif.check_fullgraph
& $py -m motif.pretrain_fullgraph --data motif/data/lipo.pt --output motif/runs/fullgraph_him_formal --method HimNet
& $py -m motif.pretrain_fullgraph --data motif/data/lipo.pt --output motif/runs/fullgraph_hmg_formal --method HMG-MRL
```

论文性质入口使用原有 `motif.train_paper` 参数，加 `--fullgraph`，并传入对应新权重。两个方法分别预训练，保留各自原始成员索引。沿用真实论文输入对齐检查、融合和预测；默认联合微调。

复用 `扩散GNN/01_扩散预训练/bond_diffusion` 的实际数据、模型、离散加噪和化学损失。五类原子属性是元素、电荷、芳香性、杂化、全图度数；键类别为无键、单、双、三、芳香。原始实现对被 mask 的键不传播消息，因此保留的是完整分子输入范围，传播使用加噪后的可见键。下游使用干净全图、t=0。

类型规则为原始成员诱导片段的规范化、带立体标记 SMILES，保留原图电荷与芳香性注释。只用训练分子的实际划分结果建立词表，验证/测试未见类型映射 unknown=1，无基序原子使用 no-motif=0。该规则是精确结构类型，粒度比官能团家族细；未做频次截断或合并，不使用此前 12 类 SMARTS。`types.json` 保存映射、训练频次和验证未知比例。

类型嵌入按原子所属基序均值聚合，加到旧模型原子嵌入上，全程不加噪。按原始成员均值聚合最终原子表示，再经可学习投影初始化论文 token/节点。投影只在下游受性质监督，全图编码器和类型嵌入受预训练监督。表示包含外部语境。

`--max-batches 1 --epochs 1 --hidden 32 --layers 2 --batch-size 2` 用于冒烟检查，权重标记 smoke，禁止冒充正式比较。验证固定随机种子且不读取测试分子。当前保存条件恢复损失，不输出新的恢复准确率；任何后续恢复指标必须标注为带基序类型条件的恢复。尚无性能增益结论。

当前入口保存 best 权重，不支持完整优化器续训。旧扩散源码通过相对工作区路径导入，迁移时需同时保留该目录。

## 本机实现核验

- `motif.check_fullgraph`：重叠均值、无基序、跨分子共享签名、外部语境、全图编码器与类型嵌入梯度、空基序均通过。
- 两套划分各完成 1 个训练批和 1 个验证批的离散扩散检查；输出在 `runs/fullgraph_him_smoke_20261016` 和 `runs/fullgraph_hmg_smoke_20261016`。
- 完整训练集建表后，HimNet 验证基序未知 170/2338，HMG-MRL 为 583/6025。该统计与冒烟训练批数独立。
- HMG-MRL 真实模型在 6 个分子、61 个基序上跑通 1 epoch 联合训练、保存和重载；见 `runs/fullgraph_hmg_paper_smoke_20261016/results.json`。保留此前有限值原版特征缓存及其来源检查。
- HimNet 首次运行通过训练和重载，但最终源码变动审计未通过，不能作为完整验收记录；固定代码后的复跑使用独立 `fullgraph_him_paper_smoke_v2_20261016` 目录。
- HimNet 固定源码复跑已完整通过：6 个分子、25 个基序，联合训练、验证选模、保存重载及最终测试均完成，`results.json` 标记 `smoke=true`。

这些检查均不是正式性质预测实验。
