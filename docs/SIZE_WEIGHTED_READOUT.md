# C/D size-weighted readout 诊断

本入口与正在运行的第一轮消融完全分离，不修改其源码、运行清单或断点。`D-no-edge` 仍由
`scripts.experiments.run_frozen_ablation` 执行；本入口只回答最终 coarse readout 的问题。

| CLI 名称 | 定义 | 配对基准 |
|---|---|---|
| `region_size_weighted` | 原 C 架构，`graph_pool` 从 `mean` 改为 `size_weighted_mean` | `region_only` |
| `base_size_weighted` | 原 D 架构，`graph_pool` 从 `mean` 改为 `size_weighted_mean` | `base_coarse` |

每个模型都从相同的 Base 母模型初始化，只切换读出；配对模型的参数、层数和初始权重完全一致。
Region/Coarse GNN 后的粗节点表示已不是原子表示的简单均值，因此这里的 size weighting 是原子量守恒的归纳偏置，
不宣称与 A 在代数上等价。

代码不会读取或产生新的 test 指标，只汇总 validation RMSE。输出使用独立目录
`outputs/frozen_size_weighted/`。

```powershell
# 只做协议、缓存、参数一致性和梯度检查
conda run --no-capture-output -n polyolefin_ml python -m scripts.readout_ablation.run_size_weighted --action prepare

# C/D 五个 seed 全部完成后，训练两个 size-weighted 版本
conda run --no-capture-output -n polyolefin_ml python -m scripts.readout_ablation.run_size_weighted --action train --seeds 0 1 2 3 4

# 也可先单独执行 C 或 D
conda run --no-capture-output -n polyolefin_ml python -m scripts.readout_ablation.run_size_weighted --action train --variants region_size_weighted --seeds 0 1 2 3 4
conda run --no-capture-output -n polyolefin_ml python -m scripts.readout_ablation.run_size_weighted --action train --variants base_size_weighted --seeds 0 1 2 3 4

# 汇总已有 validation 结果
conda run --no-capture-output -n polyolefin_ml python -m scripts.readout_ablation.run_size_weighted --action summarize
```

汇总中的差值为 `size-weighted RMSE - equal-weight RMSE`，负数表示 size-weighted 更好。
