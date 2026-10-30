"""Combine completed bounded experiments without selecting individual lucky seeds."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];module=ROOT/'motif/adjacency_diffusion';runs=module/'runs'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
for name in ['transfer_revision_20261030','posterior_weighting_20261030','posterior_bounded_20261030','task_readout_20261030','partial_transfer_20261030']:
    assert read(runs/name/'status.json')['state']=='complete'
old=read(runs/'masked_property_comparison_20261029/results.json')
regularized=read(runs/'transfer_revision_20261030/results.json')
full=read(runs/'posterior_weighting_20261030/results.json')
bounded=read(runs/'posterior_bounded_20261030/results.json')
task=read(runs/'task_readout_20261030/results.json')
partial=read(runs/'partial_transfer_20261030/results.json')
rows=[]
def add(label,m):rows.append({'label':label,'mean_rmse':m['mean_rmse'],'sample_sd':m.get('sample_sd',m.get('sd')),'n':m['n']})
add('原均值读出：从头训练',old['aggregate']['scratch'])
add('原均值读出：无遮蔽预训练',old['aggregate']['unmasked'])
add('原均值读出：遮蔽预训练',old['aggregate']['masked'])
add('较强正则：从头训练',regularized['aggregate_selected']['scratch'])
add('较强正则：遮蔽预训练',regularized['aggregate_selected']['masked'])
add('后验权重：全时间范围',full['aggregate']['posterior'])
add('后验权重：t≤300',bounded['aggregate']['posterior'])
add('新归一化＋大小读出：从头训练',task['aggregate']['scratch'])
add('新归一化＋大小读出：遮蔽预训练',task['aggregate']['masked'])
add('保留预训练主干、重置输出投影',partial['aggregate'])
best=min(rows,key=lambda r:r['mean_rmse'])
summary={'scope':'exploratory ESOL scaffold validation; all arms supervised at task stage; no test evaluation',
    'new_full_property_runs':25,'new_pretraining_runs':7,'upstream_training_molecules':512,'upstream_updates':1200,
    'property_epochs_each':60,'rows':rows,'lowest_observed_mean':best,'readout_paired_deltas':task['paired'],
    'test_evaluated':False,'limitations':['adaptive reused validation search','three seeds','single downstream dataset','no isolated boundary-gate or fixed-target context intervention']}
(module/'REVISION_RESULTS_20261030.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
lines=['# 本轮调参与设计修订结果','',
       '新完成7组预训练与25组完整60轮性质训练；另外的CPU小流程不计入性能实验。固定512分子/1200更新的预训练预算，ESOL既有scaffold验证划分。所有性质分支都使用性质监督，包括从头组。没有计算测试集指标。','',
       '| 配置 | 验证RMSE均值 ± 样本标准差（3种子） |','|---|---:|']
for r in rows:lines.append(f"| {r['label']} | {r['mean_rmse']:.5f} ± {r['sample_sd']:.5f} |")
lines+=['',f"最低观察均值为 **{best['label']}：{best['mean_rmse']:.5f}**。这是多次探索后的验证结果，不是独立测试结论。",'',
        '## 有证据支持的判断','',
        '1. 降学习率、增强正则和按观测质量归一化消息，没有改善目前的绝对最佳结果。较强正则下预训练超过相同正则的从头组，主要对应一个更弱的从头对照，不能据此宣称解决迁移。',
        '2. 原始softmax观测权重在近零正噪声下对无键赋约0.648，而t=0赋0，确有不连续性。它是启发式权重与概率解释的差异，不等于扩散公式错误。按高斯局部后验计算权重后，全范围均值仍未超过原基线；不能把这个差异当成已证实的退化原因。',
        '3. 保持目标完全遮蔽、只限制周围噪声到t≤300，也未得到稳定改善。较高或较低恢复率都不是直接优化目标。',
        '4. 当前18通道的训练分子诊断显示初始化与预训练图表示尺度明显不同，但这只是读出适配的动机；新读出的归一化和原子数特征作为一个组合验证，不能分别归因。','',
        '## 新读出的严格配对比较','', '| 种子 | 从头训练 | 预训练 | 预训练减从头 |','|---:|---:|---:|---:|']
for seed in [0,1,2]:
    pair={r['arm']:r['validation']['rmse'] for r in task['runs'] if r['seed']==seed}
    lines.append(f"| {seed} | {pair['scratch']:.5f} | {pair['masked']:.5f} | {pair['masked']-pair['scratch']:+.5f} |")
delta=task['aggregate']['masked']['mean_rmse']-task['aggregate']['scratch']['mean_rmse']
absolute=task['aggregate']['masked']['mean_rmse']-old['aggregate']['scratch']['mean_rmse']
lines+=['',f'新读出下预训练相对同架构从头组的均值差为 {delta:+.5f}；相对原最强从头均值的差为 {absolute:+.5f}。负数更好。必须同时看这两个比较，不能把新架构或被削弱的基线造成的差异全部归因于去噪。','',
        '只重置输出投影的独立对照保持原架构和容量，详细逐种子差值见 [部分迁移](PARTIAL_TRANSFER_20261030.md) 及对应运行目录；不能把投影重置的效应和读出候选混为一谈。','',
        '相关接口/模式测试通过；所有完整训练检查点回放、冻结源码哈希与配对初始化验收通过。权重留在本地，文件哈希随清单提供。失败候选、监督分母、梯度裁剪记录和逐种子曲线均保留。','',
        '这些结果没有证明门控的独立贡献、严格缺失基序推断、完整分子精确恢复或其他数据集收益。验证集被多轮使用，当前均值含选择乐观性；不能将微小差异写成已证实的普遍增益。','',
        '详细协议：[调参与聚合](TRANSFER_REVISION_20261030.md)、[观测权重与噪声范围](POSTERIOR_WEIGHTING_20261030.md)、[任务读出](TASK_READOUT_20261030.md)。']
(module/'REVISION_RESULTS_20261030.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
readme=(module/'README.md').read_text(encoding='utf-8')
historical=readme[readme.index('本模块与 `brics_atom_diffusion`'):].replace('`n固定18','固定18')
prefix=['# 化学边恢复与分子性质预测：实验实现','',
        '2026-10-30：本轮7组预训练、25组完整性质训练及相关接口测试完成。所有数值均为ESOL验证结果；没有评估测试集。完整候选和失败结论见 [修订结果](REVISION_RESULTS_20261030.md)。','',
        '整体审视优先于继续搜索：本轮主要排除了局部配置，尚未闭合“恢复任务→干净表示→性质预测”的证据链；停止新增训练，见 [整体方法审计](METHOD_AUDIT_20261030.md)。','',
        f"目前最低观察均值：{best['label']}，RMSE {best['mean_rmse']:.5f}。这是多轮探索后的验证均值，不能据此宣称普遍或显著增益。",'',
        '保留全部原子和键属性。默认旧均值读出、候选数归一化和softmax权重不被失败候选覆盖；新读出通过独立 `TaskReadoutModel` 和配对运行脚本提供。额外模式与协议见 [任务读出](TASK_READOUT_20261030.md)、[调参/聚合](TRANSFER_REVISION_20261030.md)、[权重/噪声范围](POSTERIOR_WEIGHTING_20261030.md)。','',
        '此前固定18通道三组均值为：从头0.78711、无遮蔽预训练0.79606、遮蔽预训练0.81446。目标完全遮蔽后，准确任务是在高斯受损周围环境中恢复未知化学边。旧测试及性质协议见 [遮蔽任务](MASKED_RECOVERY_20261029.md) 和 [三组对照](MASKED_PROPERTY_PROTOCOL_20261029.md)。','',
        '## 以下为早期二值实现及历史验收记录','']
(module/'README.md').write_text('\n'.join(prefix)+'\n'+historical,encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False))
