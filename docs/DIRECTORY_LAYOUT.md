# 目录与命令索引

所有命令均在 `coarse_graining` 根目录执行，使用 `polyolefin_ml` 环境。
迁移后的 Python 工具采用 `python -m 包名.模块名`，使项目内部导入可被正确解析。

## 文件放置规则

| 位置 | 放什么 |
|---|---|
| 根目录三个训练文件 | 当前正式实验入口及冻结特征支持代码 |
| `coarse_gnn/` | 粗化、Region/Coarse GNN、缓存和批处理等核心实现 |
| `diffusion/` | 原始扩散子项目、数据、权重及其独立说明 |
| `examples/` | 使用预训练编码器的最小演示 |
| `scripts/data/` | 拓扑预计算等数据准备工具 |
| `scripts/experiments/` | 第一轮核心消融的统一模型开关、原训练器接入与汇总 |
| `scripts/validation/` | 重编号不变性和数值等价性检查 |
| `scripts/performance/` | 性能分析和隔离的断点测速 |
| `scripts/launchers/` | Windows 后台训练启动脚本 |
| `scripts/maintenance/` | 可选的打包、上传准备工具 |
| `tests/` | 自动化测试 |
| `docs/` | 实验协议和项目使用说明 |
| `outputs/` | 本地演示、方法检查、性能报告和拓扑缓存 |

正式训练输出继续位于项目旁的 `../model/results_formal/05_coarse_gnn/`，Frozen 实验位于其 `frozen_mechanism/` 子目录。旧输出、检查点和历史源码快照沿用原位置。

## 常用命令

第一轮核心消融仅准备代码，入口及后续命令见 [第一轮实验说明](ABLATION_ROUND1.md)。

```powershell
# 冻结实验入口的参数说明（不启动训练）
conda run --no-capture-output -n polyolefin_ml python run_lipo_frozen.py --help

# CPU 演示及一次梯度检查
conda run --no-capture-output -n polyolefin_ml python -m examples.run_coarse_demo --variant base --backward --output outputs/coarse_demo/new_smoke.json

# 拓扑预计算
conda run --no-capture-output -n polyolefin_ml python -m scripts.data.precompute_topology --smiles 'CCO' 'CC(=O)O' --cache-dir outputs/topology_cache

# 方法不变性检查
conda run --no-capture-output -n polyolefin_ml python -m scripts.validation.audit_method --permutations 100 --variant base --output outputs/method_audit/new_audit.json

# 自动化测试
conda run --no-capture-output -n polyolefin_ml python -m unittest discover -s tests -v
```

需要启动或恢复正式训练时，按 [Frozen 协议](FROZEN_MECHANISM.md) 执行。后台脚本位置分别为 `scripts/launchers/start_lipo_frozen.ps1` 和 `scripts/launchers/start_lipo_optimized.ps1`，内部会自动切换到项目根目录；原有 seed 和输出位置保持不变。`optimized` 对应暂停的 finetune 计划。

性能工具使用 GPU 或真实断点，应在需要测速时单独运行：

```powershell
conda run --no-capture-output -n polyolefin_ml python -m scripts.performance.profile_coarse
conda run --no-capture-output -n polyolefin_ml python -m scripts.performance.benchmark_resumed_epoch
conda run --no-capture-output -n polyolefin_ml python -m scripts.validation.verify_packed
```

## 旧文件位置 → 新文件位置

| 原根目录文件 | 新位置 |
|---|---|
| `run_coarse_demo.py` | `examples/run_coarse_demo.py` |
| `precompute_topology.py` | `scripts/data/precompute_topology.py` |
| `audit_method.py` | `scripts/validation/audit_method.py` |
| `verify_packed.py` | `scripts/validation/verify_packed.py` |
| `profile_coarse.py` | `scripts/performance/profile_coarse.py` |
| `benchmark_resumed_epoch.py` | `scripts/performance/benchmark_resumed_epoch.py` |
| `prepare_github_upload.py` | `scripts/maintenance/prepare_github_upload.py` |
| `start_lipo_frozen.ps1` | `scripts/launchers/start_lipo_frozen.ps1` |
| `start_lipo_optimized.ps1` | `scripts/launchers/start_lipo_optimized.ps1` |
| `FROZEN_MECHANISM.md` | `docs/FROZEN_MECHANISM.md` |
| `LIPO_FORMAL.md` | `docs/LIPO_FORMAL.md` |
| `GITHUB_UPLOAD.md` | `docs/GITHUB_UPLOAD.md` |

## 为什么保留三个根目录训练文件

2026-09-15 整理时，`run_lipo_frozen.py --action train --seeds 2 3 4` 正在执行。
`run_lipo_frozen.py`、`run_lipo_formal.py`、`frozen_features.py` 的原路径及内容哈希属于现有实验协议。
因此这三个文件集中作为稳定训练入口保留，避免目录整理改变实验身份、影响后续断点恢复。
整理没有修改这三个文件或 `coarse_gnn/*.py`，没有改写实验配置、数值审计 JSON、权重和历史结果。
后续新工具按上表分类放置，不再堆到根目录。
