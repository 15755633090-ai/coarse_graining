import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];module=ROOT/'motif/adjacency_diffusion';out=module/'runs/representation_matrix_20261030'
r=json.loads((out/'results.json').read_text(encoding='utf-8'))
assert json.loads((out/'status.json').read_text(encoding='utf-8'))['state']=='complete'
p=module/'REPRESENTATION_MATRIX_20261030.md';s=p.read_text(encoding='utf-8').split('\n## 首轮完整结果')[0]
s=s.replace('状态：代码和接口验收完成，正式运行以`runs/representation_matrix_20261030/status.json`为准。没有预先宣称性质增益。',
    '状态：首轮四格对照全部完成。12组来源/12组性质结果中复用原3组，实际新增9组预训练和9组完整60轮性质训练；来源及回放验收通过，没有测试集评估。')
s+='\n## 首轮完整结果\n\n| 语料 | 目标 | 验证RMSE均值 ± 样本标准差（3种子） |\n|---|---|---:|\n'
for key,m in r['aggregate'].items():
    corpus,obj=key.split('_',1);s+=f"| {corpus} | {obj} | {m['mean_rmse']:.5f} ± {m['sample_sd']:.5f} |\n"
s+='''
原同架构scratch为0.78711 ±0.02804。新目标在512条件下三个种子均差于旧目标；4096条件下三个种子均优于同语料旧目标，但仍未超过原最强scratch均值，也未超过原512边恢复均值。这个条件差异不代表瓶颈已突破，未等算力匹配时也不能单归因于目标公式。

新目标六个来源在128个旧恢复确认分子的干净图读出上，有效秩均约1.0001到1.0004，维度标准差约0.68到0.76，并非常量输出。旧目标同一诊断约2.61到3.74。低秩本身不能证明性质无用，但与本轮未获收益一起，否定了用很小视图误差来选择此配方的依据。没有证明所有原子隐状态也为一维，或唯一原因已定位。

4096语料的优化批次实际见到3668到3711个不同分子；全部4096参与无标签类别先验统计。同源ZINC粗结构分布仍不同于ESOL训练域，数量扩展不是领域对齐。后续若重新设计，应先论证视图保留的信息、目标表征质量、防退化机制的小批次适用性与训练域覆盖。本轮不继续调正则权重或恢复噪声参数，也不把它当成对LeJEPA论文或所有表示预训练的否定。

全部配对差值、交互项、GPU计算事件时间、实际曝光和有效秩记录见运行目录REPORT.md、results.json、sources.json。权重留在本地，SHA256清单随同步提供；冻结源码以representation_sources_20261030.tar.xz归档。
'''
p.write_text(s,encoding='utf-8')
p=module/'README.md';text=p.read_text(encoding='utf-8');title,rest=text.split('\n',1)
prefix='表示目标×语料首轮四格对照已完成：512边恢复0.81446、512表示一致性0.83449、4096边恢复0.84288、4096表示一致性0.82339；均未超过原scratch0.78711。保留原默认，不追加相邻参数搜索。实现、退化诊断和范围见 [表示目标与语料对照](REPRESENTATION_MATRIX_20261030.md)。'
if prefix not in text:p.write_text(title+'\n\n'+prefix+'\n'+rest,encoding='utf-8')
print(json.dumps({'summarized':True,'aggregate':r['aggregate'],'no_test_evaluation':True}))
