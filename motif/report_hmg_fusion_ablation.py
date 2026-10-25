"""Standalone, standard-library-only audit and report for HMG fusion ablations."""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def rows(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True)
    root=p.parse_args().results
    protocol=json.loads((root/'protocol.json').read_text(encoding='utf-8'))
    runs=rows(root/'runs.csv');summary=rows(root/'summary.csv')
    assert len(runs)==21
    max_error=0.
    for run in runs:
        folder=root/f"seed_{run['seed']}"/run['arm']
        assert (folder/'best.pt').is_file()
        for split in ('valid','test'):
            predictions=rows(folder/f'{split}_predictions.csv')
            assert [int(r['row']) for r in predictions]==protocol['rows'][split]
            value=math.sqrt(statistics.mean((float(r['prediction'])-float(r['target']))**2 for r in predictions))
            error=abs(value-float(run[f'{split}_rmse']));assert error<1e-6
            max_error=max(error,max_error)
    paired={}
    for row in rows(root/'paired_deltas.csv'):
        paired.setdefault((row['combo'],row['control']),[]).append(float(row['delta_rmse']))
    lines=['# HMG-MRL 原融合协议七组消融','',
        '沿用此前原架构的统一监督训练协议：原 Lipo 划分、seed 0/1/2、原输入和原融合算子、'
        '原预测头、原辅助对齐目标。不是原论文全部预训练及实验流程的复现。','',
        'A=原子分支；S=Motif 分支；D=描述符分支。D 并非 HimNet 的全图节点，不跨论文直接比较尺度含义。','',
        '每组保留完整分支计算和 A/S 辅助对齐，只改变参与预测融合的视图，属于预测融合消融，'
        '不是结构分支删除。所有模型从相同种子的原始初始化开始联合训练；ASD 复用已验收原 Full，'
        '其余 18 次为新增训练。复用时重载验证，不重新执行历史测试。','',
        '| 组别 | 测试 RMSE 均值 ± 样本标准差 |','|---|---:|']
    for r in summary:lines.append(f"| {r['arm']} | {float(r['test_rmse_mean']):.4f} ± {float(r['test_rmse_sd']):.4f} |")
    lines+=['','## 配对比较','','Δ=RMSE(组合)−RMSE(对照)，负值较好。区间仅为固定划分、三个训练种子的描述性 95% t 区间，n=3 不稳定，未经多重比较校正。','',
        '| 组合 | 对照 | 平均 ΔRMSE | 95% 区间 | 改善次数 |','|---|---|---:|---|---:|']
    for (combo,control),values in paired.items():
        assert len(values)==3
        avg=statistics.mean(values);radius=4.30265273*statistics.stdev(values)/math.sqrt(3)
        lines.append(f'| {combo} | {control} | {avg:+.4f} | [{avg-radius:+.4f}, {avg+radius:+.4f}] | {sum(v<0 for v in values)}/3 |')
    capped=sum(int(r['epochs_run'])==100 for r in runs)
    lines+=['',f'{capped}/21 个结果达到原定 100 epoch 上限，不能宣称全部充分收敛。','',
        '融合层和预测头不随视图数加宽，注册参数总数相同；未参与损失路径的参数不等于有效训练容量。'
        '各新组 gradient_audit.json 报告实际梯度路径，未声称已匹配有效参数容量。','',
        '本轮保留原 HMG 的构象相关键特征与描述符输入，以及已有训练集统计填补后的有限值缓存。'
        '历史测试结果已查看；本轮不据测试结果追加预算或调参。','',
        'ASD 保持原 Full 计算图；此结果与约 0.95 的 HimNet 冻结读出探针分表，不能混称同一实验协议。']
    (root/'REPORT_CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (root/'artifact_audit.json').write_text(json.dumps(dict(passed=True,checkpoints=21,prediction_files=42,
        max_metric_recomputation_error=max_error,model_test_rerun=False),indent=2),encoding='utf-8')
    print('\n'.join(lines[:18]))


if __name__=='__main__':main()
