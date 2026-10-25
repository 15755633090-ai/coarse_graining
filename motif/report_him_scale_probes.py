"""Summarize a completed fixed-encoder experiment without rerunning predictions."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def csv_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    a = parser.parse_args(); root = a.results
    completion = json.loads((root/'completion.json').read_text(encoding='utf-8'))
    protocol = json.loads((root/'protocol.json').read_text(encoding='utf-8'))
    assert completion['passed']
    runs = csv_rows(root/'runs.csv'); summary = csv_rows(root/'summary.csv')
    assert len(runs) == completion['runs'] == len(protocol['configs'])*len(protocol['head_seeds'])
    for name, digest in protocol['implementation_sha256'].items():
        assert sha(root/name) == digest
    # Recompute metrics from saved predictions, without another model evaluation.
    max_metric_error = 0.
    for run in runs:
        folder = root/f"seed_{run['seed']}"/run['name']
        assert (folder/'best.pt').is_file()
        for split, metric in (('valid', 'valid_rmse'), ('test', 'test_rmse')):
            predictions = csv_rows(folder/f'{split}_predictions.csv')
            assert [int(r['row']) for r in predictions] == protocol['rows'][split]
            calculated = math.sqrt(statistics.mean(
                (float(r['prediction'])-float(r['target']))**2 for r in predictions))
            assert math.isfinite(calculated)
            error = abs(calculated-float(run[metric])); max_metric_error = max(max_metric_error, error)
            assert error < 1e-6
    (root/'artifact_audit.json').write_text(json.dumps(dict(passed=True,
        checkpoints=len(runs), prediction_files=2*len(runs),
        rows_per_prediction_file={s: len(protocol['rows'][s]) for s in ('valid', 'test')},
        max_metric_recomputation_error=max_metric_error, model_test_rerun=False), indent=2), encoding='utf-8')
    groups = {}
    for r in csv_rows(root/'paired_deltas.csv'):
        groups.setdefault((r['combo'], r['control']), []).append(float(r['delta_rmse']))
    paired = []
    for (combo, control), values in groups.items():
        avg = statistics.mean(values); sd = statistics.stdev(values)
        # Fixed five-seed descriptive interval, conditional on the one encoder/split.
        assert len(values) == 5
        radius = 2.7764451052*sd/(len(values)**.5)
        paired.append(dict(combo=combo, control=control, n=len(values),
                           delta_mean=avg, delta_sd=sd, ci95_low=avg-radius,
                           ci95_high=avg+radius, wins=sum(v < 0 for v in values)))
    with (root/'paired_summary.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(paired[0])); w.writeheader(); w.writerows(paired)
    lookup = {r['name']: r for r in summary}
    lines = ['# HimNet 三尺度冻结读出：第一轮结果', '',
        '固定修正版 Full seed 0 编码器，在原 Lipo 3360/420/420 划分上比较预测头。'
        '五个种子仅改变预测头训练；不是五个编码器，也不是五个独立划分。', '',
        f"共完成 {len(runs)} 次训练；每次按验证集选模、严格重载后测试一次。"
        '编码器来源训练使用过指纹和辅助对齐；本轮预测头不接入指纹。历史测试结果已被查看。', '',
        '## 主组', '', '| 组别 | 预测头参数 | 测试 RMSE 均值 ± 标准差 |', '|---|---:|---:|']
    reference_root = Path(protocol['checkpoint']).parent
    reference = json.loads((reference_root/'full_test_predictions.json').read_text(encoding='utf-8'))
    assert reference['row_ids'] == protocol['rows']['test']
    current = csv_rows(root/'seed_0/ASG/test_predictions.csv')
    assert all(float(new['target']) == old for new,old in zip(current, reference['target']))
    reference_rmse = math.sqrt(statistics.mean((p-y)**2 for p,y in zip(reference['prediction'],reference['target'])))
    insert = ['## 与历史约 0.7 的结果区别', '',
        f'同一来源 checkpoint 的原完整 HimNet（seed 0）测试 RMSE 为 **{reference_rmse:.6f}**。'
        '该数值由已保存测试预测重算，没有重新执行模型测试。测试行号与标签和本轮一致。', '',
        '| 项目 | 原完整模型 | 本轮 ASG 冻结探针 |', '|---|---|---|',
        '| 原子/Motif/全图编码 | 原有双路径 | 固定同一训练后编码器 |',
        '| 图读出 | 全部节点按数量平均，再输出映射 | 各类节点先平均并分别映射，再拼接 |',
        '| 指纹预测支路 | 保留 | 不接入预测头 |',
        '| 融合与预测头 | 原有融合和联合训练头 | 新 MLP 重新训练 |',
        '| 训练方式 | 完整模型联合训练 | 仅训练预测头 |', '',
        '因此 0.9461 不是原完整 HimNet 的复现值，也不能把 0.7144→0.9461 的差异全部归因于冻结或删除指纹。'
        '本轮改变了多项预测端条件，缺少逐项匹配对照。三尺度分组名称相同不代表模型计算图相同。', '',
        '此前汇总遗漏这条原模型参考；现补充此对照边界。已完成的 65 次运行保留为冻结读出探针，不能改称原模型七组消融。', '']
    lines[6:6] = insert
    for name in ('A', 'S', 'G', 'AS', 'AG', 'SG', 'ASG'):
        r = lookup[name]
        lines.append(f"| {name} | {r['parameters']} | {float(r['test_rmse_mean']):.4f} ± {float(r['test_rmse_sd']):.4f} |")
    lines += ['', '## 容量匹配', '',
        '双尺度和三尺度各设三个单尺度对照；同预算、同配置的对照在不同组合比较中复用。', '',
        '| 单尺度对照 | 预测头宽度 | 参数 | 测试 RMSE 均值 ± 标准差 |', '|---|---:|---:|---:|']
    for r in summary:
        if r['control'] != 'main':
            lines.append(f"| {r['name']} | {r['width']} | {r['parameters']} | {float(r['test_rmse_mean']):.4f} ± {float(r['test_rmse_sd']):.4f} |")
    lines += ['', '## 配对差值', '',
        'Δ = RMSE(组合) − RMSE(对照)，负值为组合较好。区间为五个配对预测头种子的描述性 95% t 区间，'
        '仅覆盖当前编码器、当前划分下的训练波动；未经多重比较校正，不代表跨划分或物理机制证据。'
        'ASG 对双尺度的比较尚未设置容量匹配的双尺度对照，只作描述，不能单独归因为第三尺度的信息。', '',
        '| 组合 | 对照 | 平均 ΔRMSE | 95% 区间 | 组合较好次数 |', '|---|---|---:|---|---:|']
    for r in paired:
        lines.append(f"| {r['combo']} | {r['control']} | {r['delta_mean']:+.4f} | [{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}] | {r['wins']}/5 |")
    lines += ['', '## 如何解释', '']
    for combo in ('AS', 'AG', 'SG', 'ASG'):
        rr = [r for r in paired if r['combo'] == combo and r['control'] not in ('AS','AG','SG')]
        against_all = all(r['delta_mean'] < 0 for r in rr)
        every_seed = all(r['wins'] == 5 for r in rr)
        lines.append(f"- {combo}：相对全部组成单尺度及对应容量对照，平均误差{'均更低' if against_all else '并非均更低'}；"
                     f"各比较{'均为五个种子一致改善' if every_seed else '未全部达到五个种子一致改善'}。")
    capped = sum(r['reached_budget'] == 'True' for r in runs)
    lines += ['', f'有 {capped}/{len(runs)} 次训练到达预设 100 epoch 上限；完整轮次与最佳轮次见 runs.csv。'
        '本轮不依据测试结果延长预算或追加调参。', '',
        f"无 Motif 分子计数：{completion['no_motif']}。无 Motif 时 S 固定为零向量。", '',
        'A/S/G 来自已经完成跨尺度交互的节点状态；A 不是删除其他节点后的原子模型。'
        '本轮结论限于节点尺度输出的预测价值与互补性。整体节点表示、原子汇聚和 Motif 汇聚不应被解释为彼此独立的信息来源。', '',
        '结果由已保存预测与指标汇总，不重新执行模型测试。']
    (root/'REPORT_CN.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, constrained_layout=True)
        for ax, names, title in (
            (axes[0], ['A','S','G','AS','AG','SG','ASG'], 'Main readouts'),
            (axes[1], [r['name'] for r in summary if r['control'] != 'main'], 'Capacity-matched controls')):
            means = [float(lookup[n]['test_rmse_mean']) for n in names]
            sd = [float(lookup[n]['test_rmse_sd']) for n in names]
            ax.errorbar(range(len(names)), means, yerr=sd, fmt='o', capsize=4)
            ax.set_xticks(range(len(names)), [n.replace('_matched_', '\nmatched ') for n in names])
            ax.set_ylabel('Test RMSE (lower is better)'); ax.set_title(title)
            ax.grid(axis='y', alpha=.25)
        fig.suptitle('Frozen repaired HimNet: one encoder, five head seeds; error bars = SD')
        fig.savefig(root/'readout_comparison.png', dpi=180); plt.close(fig)
    except ImportError:
        pass
    print('\n'.join(lines[:22]))


if __name__ == '__main__':
    main()
