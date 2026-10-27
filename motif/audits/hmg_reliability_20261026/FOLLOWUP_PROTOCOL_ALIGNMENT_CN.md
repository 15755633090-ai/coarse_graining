# 继续对标论文：回归参数、公开代码边界与尚缺证据

日期：2026-10-26。正式实验继续暂停。本轮核对正文全文、16 页补充材料、官方唯一公开分支及两个公开提交；只运行修改验收，不进行正式测试或调参。

## 已解决的新差异：回归温度

[补充材料第 5 节](https://academic.oup.com/bioinformatics/article/42/8/btag600/8758503#supplementary-data)明确按任务类型选择辅助目标参数：分类 alpha=0.15、tau=0.15，回归 alpha=0.15、tau=0.30，并说明回归设置适用于所有回归数据集。Lipo 因此应使用 tau=0.30，而不能继承分类 config.py 中的 0.15。

现已在 v2 模型构建前显式应用回归参数，保持官方 info_nce_loss 函数不变。protocol.json 和每组验收记录实际温度，并在复用新 Full 基线时核对温度与辅助系数。

独立数值验收：实际官方 InfoNCE=2.7351317405700684，与手算 tau=0.30 完全相同；相同输入下 tau=0.15 为 3.9824275970458984，确认运行值实际改变，而非只改配置日志。梯度有限；该数值来自固定随机特征，不是分子性能结果。

既有 v1 和上一轮 v2 冒烟均继承过 0.15，不能继续称为回归参数已全部对齐。前述 v2 冒烟保留历史记录，新的 tau=0.30 七组验收单独保存；不重新解释旧性能排名，也不将偏差直接归因为温度。

## 公开代码已明确，但不能推广成回归确认的设置

[官方固定版本](https://github.com/AKZstar/HMG-MRL/tree/6ad69261eb1af36c8c0b96594d79a7639b969223)的 config.py 和 run_main.py 给出：

- epochs=100、max_epochs=100。
- Noam 调度，warmup=2 epoch，init_lr=1e-4、max_lr=1e-3、final_lr=1e-4；训练 batch 后调用 scheduler.step。
- 分类早停实际是 ROC 停滞超过 50 轮且 loss 停滞超过 20 轮同时满足，不是直接的回归 RMSE patience=50。
- grid 中 seed=2020–2024，名义 learning_rate=1e-3；config.py 的静态 learning_rate=5e-4 会被 grid 覆盖，之后 Noam 还会改变实际学习率。

但该入口使用分类标签、加权 CrossEntropyLoss、两类输出和 ROC 选模，没有可直接用于 Lipo 的回归入口。不能把“100 轮、Noam”当成已证明的论文回归设置。

## 各项目前能确认到什么程度

| 项目 | 正文/补充材料 | 公开代码 | 当前决定 |
|---|---|---|---|
| 回归损失 | 明确 MSE | 入口是分类交叉熵 | MSE；目标是否缩放仍未说明 |
| 回归 alpha / tau | 补充第 5 节明确 0.15 / 0.30 | 分类默认 0.15 / 0.15 | v2 修正为 0.15 / 0.30 |
| 最大轮数 | 正文未给上限；只明确 patience=50 | 分类配置 100 | 回归值不自动补 100 |
| 学习率调度 | 初始 1e-3；未给调度细节 | 分类 Noam | constant 是显式假设，不称论文确认；未擅自套分类 Noam |
| 目标缩放 | 全文检索未找到标签缩放说明 | 无回归训练入口 | 保留显式选择，不能因未提及就断言没有缩放 |
| 五份 scaffold 行号 | 8:1:1、五次运行 | 外部 scaffold_split 缺失，无 fold CSV | v2 新行号不冒称作者划分 |
| 单粒度变体 | 补充第 12 节及表 S17/S18 | 未发布相应变体入口 | 不将融合子集消融冒称作者单粒度实验 |

## 公开材料范围检查

远端 main 仍为 6ad6926。当前仅一个公开分支；公开历史只有初始发布 ba9dabee 和 README 修改 6ad6926，二者仅 README 不同；没有其他训练版本可从这段公开历史恢复。查询时 issue 与 release 列表均为空。tags 端点不被当前连接器支持，因此不声明已核查 tags。

论文正文与补充材料已下载并校验文件哈希，全文提取 53,196 字符、补充材料 16 页，所有 epoch/调度和标签缩放相关命中记录在 public_materials_search.json。关键词未命中只是辅助检索证据，不能证明作者私有流程一定没有对应步骤。

## 另外需要作者澄清的公式/源码表述

- 正文式 (11) 展示加权和；公开 forward 还执行视图求和残差和激活。当前继续保留官方 forward，不擅自删残差来追求字面公式。
- 正文称 learnable CLS；公开预处理首 token 由 whole_mol 原子/键统计初始化，随后经过可学习投影及注意力。需确认“learnable CLS”的实现含义，不能擅自加一个自由 nn.Parameter。

这些是待澄清的表述/实现对应，不认定为作者方法错误，更不是当前排名的因果解释。

## 可复核记录

- protocol_evidence_matrix.json：9 项设置状态与来源。
- public_repository_scope.json：分支、提交、历史差异、issue/release 查询。
- public_materials_search.json：正文和补充材料搜索及下载哈希。
- supplement_page_6.png：回归 alpha/tau 原文页视觉核对；第 13 页同时核对单粒度/参数匹配表。
- motif/runs/hmg_protocol_v2_regression_temperature_20261026/acceptance.json：实际 InfoNCE 数值检查。
- motif/runs/hmg_protocol_v2_smoke_tau030_20261026/acceptance.json：修正后七组真实分子小样本流程验收，以该文件存在且 passed=true 为准。

下一步要准确复现论文回归流程，需要作者回归配置/目标处理与五份划分，或明确采用有标识的工程假设及敏感性验证；不能把公开材料缺口写成已解决。
