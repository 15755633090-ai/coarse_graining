import json
import argparse
from pathlib import Path
import statistics
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--run',type=Path,default=ROOT/'motif/adjacency_diffusion/runs/typed_recovery_comparison_20261029')
run=parser.parse_args().run
status=json.loads((run/'status.json').read_text(encoding='utf-8'))
if status.get('state')!='complete':
    raise RuntimeError('Wait for all planned matched-budget cells')
rows=json.loads((run/'results.json').read_text(encoding='utf-8'))['runs']
groups={}
for row in rows:
    groups.setdefault(row['variant'],[]).append(row)
output={}
for variant,group in groups.items():
    output[variant]={'seeds':[r['seed'] for r in group]}
    for fold in ('development','confirmation','challenge'):
        output[variant][fold]={}
        for t in ('50','200','500','1000'):
            measured=[r[fold]['by_timestep'][t] for r in group]
            means={key:statistics.mean(values) if values else None
                   for key in measured[0] if key!='counts'
                   for values in [[r[key] for r in measured if r[key] is not None]]}
            p,r=means['presence_precision'],means['presence_recall']
            means['presence_F1_from_mean_precision_recall']=2*p*r/(p+r) if p is not None and r is not None and p+r else 0.
            for label,key in [('type_accuracy_among_detected_true_bonds','type_bond_correct'),
                              ('joint_accuracy_among_detected_true_bonds','joint_bond_correct')]:
                values=[r['counts'][key]/r['counts']['true_positive'] for r in measured if r['counts']['true_positive']]
                means[label]=statistics.mean(values) if values else None
            output[variant][fold][t]=means
baseline=json.loads((run/'observation_only.json').read_text(encoding='utf-8'))
aggregate={'groups':output,'zero_parameter_observation_baseline':baseline,'test_evaluated':False,
           'unique_confirmation_molecules':128,'means_are_across_two_seeds_not_independent_molecule_replications':True}
(run/'aggregate.json').write_text(json.dumps(aggregate,indent=2),encoding='utf-8')
for variant,row in output.items():
    m=row['confirmation']['500']
    print(json.dumps({'variant':variant,'AP500':m['presence_AP'],'type500':m['type_accuracy_true_bonds'],
                     'joint500':m['joint_accuracy_true_bonds'],'aromatic500':m['aromatic_accuracy'],
                     'double500':m['double_accuracy'],'precision500':m['presence_precision'],'recall500':m['presence_recall']}))
