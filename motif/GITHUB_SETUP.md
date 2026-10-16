# 从 GitHub 使用本模块

2026-10-16 更新：全图条件扩散的正式 seed-0 实验、日志、预测及压缩权重/输入缓存已收录在 `results/fullgraph_formal_seed0_20261016/`，恢复方式见其中 README。下方不含运行产物的说明适用于历史孤立基序路径。

本目录是工作区 `motif/` 的源码快照。当前提交不包含数据集、126 MB 的输入缓存、模型权重、运行输出，也不包含两篇论文的整个第三方仓库。

`README.md` 保留原工作区运行记录与示例路径；独立克隆本仓库时，请使用下面的显式输入路径，不依赖工作区层级或本机 Python 路径。

## 独立运行

在本仓库根目录执行，使用 Python 3.9 或更新版本：

```bash
python -m pip install -r motif/requirements.txt
python -m motif.checks

# 输入来自已有的结构审计；此命令不重新提取或修改化学基序。
python -m motif.prepare --audit /path/to/verified_motifs.json --labels /path/to/graph-label.csv.gz --output motif/data/lipo.pt

python -m motif.pretrain --data motif/data/lipo.pt --output motif/runs/pretrain_seed0 --epochs 50 --seed 0
python -m motif.train_property --data motif/data/lipo.pt --checkpoint motif/runs/pretrain_seed0/best.pt --output motif/runs/property_hmg_seed0 --method HMG-MRL --seed 0
```

`verified_motifs.json` 必须使用本项目的已审计格式：每条记录包含 `source_row`、`smiles`、`split`、`method`、`motifs`，每个 motif 有 `atoms` 成员索引。`source_row` 对应标签文件的零起始行号。相邻 `manifest.json` 存在时，准备脚本会核验结构数据 SHA-256 与未解决结构错误数。

本地当前完整输入的结构数据 SHA-256：`5e8bcdb391be744633ffb64c8d1f5cc503769b8cdf4bd4921bcc13981760176b`。实际缓存信息及特征说明见原工作区的 `motif/data/lipo.json`；缓存不是仓库内容。

## 论文适配器

`adapters.py` 接受调用方已经构建好的原模型对象和成员映射；使用前按 README 的 API 传入这些对象。适配器本身不下载原项目，不修改其源码。

可参考原始仓库：

- HMG-MRL：https://github.com/AKZstar/HMG-MRL
- HimNet：https://github.com/Hugh415/HimNet

`check_paper_adapters.py` 是原工作区集成检查脚本，其路径依赖原工作区的两套论文代码与既有 Windows 兼容副本，不能在缺少这些依赖的干净克隆上直接执行。独立的 `motif.checks` 不需要这两套仓库。

修订后的 `train_paper.py`、`prepare_hmg.py`、`check_paper_data.py` 均通过 `--paper-root` 接受调用方的原项目路径，不依赖固定工作区层级。完整命令和环境说明见 [PAPER_TRAINING.md](PAPER_TRAINING.md)。`train_property.py` 仍是独立语境网络，不能用它的结果声称原论文＋模块收益。

HMG-MRL 官方未提供现成特征缓存；`prepare_hmg.py` 复用官方 save_smiles_dicts 生成，保留其原始构象和描述符依赖。读取原缓存无需安装 DeepChem，生成则需要 DeepChem 2.8.0 及原预处理环境；可参考 requirements_paper.txt。不要把本机 vendor 目录上传到仓库。

内部预训练只使用二维数据；原模型其他分支的特征与依赖由原模型决定。当前已完成修订后的实现检查，以及两套原模型各 6 个真实分子的完整小规模运行；尚未进行正式效果训练或建立预测收益。
