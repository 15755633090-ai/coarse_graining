"""Summarize all matched seeds; never select on the confirmation group."""
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
run=ROOT/'motif/adjacency_diffusion/runs/posterior_comparison_20261029'
status=json.loads((run/'status.json').read_text(encoding='utf-8'))
if status.get('state')!='complete':
    raise RuntimeError('All eight cells must complete before aggregate conclusions')
rows=json.loads((run/'results.json').read_text(encoding='utf-8'))['runs']
groups={}
for row in rows:
    groups.setdefault(row['variant']['id'],[]).append(row)
output={}
for name,group in groups.items():
    output[name]={
        'seeds':[r['seed'] for r in group],
        'confirmation_mid_ratio':statistics.mean((r['confirmation']['200']['density_ratio']+r['confirmation']['500']['density_ratio'])/2 for r in group),
        'confirmation_AP500':statistics.mean(r['confirmation']['500']['average_precision'] for r in group),
        'confirmation_AP1000':statistics.mean(r['confirmation']['1000']['average_precision'] for r in group),
        'confirmation_recall1000':statistics.mean(r['confirmation']['1000']['positive_recall'] for r in group),
        'confirmation_precision1000':statistics.mean(r['confirmation']['1000']['precision'] for r in group)}
(run/'aggregate.json').write_text(json.dumps({'groups':output,'test_evaluated':False,'selection':None},indent=2),encoding='utf-8')
print(json.dumps(output))
