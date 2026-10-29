"""Retain small-group stereochemical diagnostics for all six source encoders."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
run=ROOT/'motif/adjacency_diffusion/runs/typed_property_comparison_20261029'
sources=json.loads((run/'resolved_pretrain_sources.json').read_text(encoding='utf-8'))
rows=[]
for seed,arms in sources.items():
    for arm,item in arms.items():
        folder=Path(item['path']).parent
        original=folder/'result.json'
        if original.exists():
            measurements=json.loads(original.read_text(encoding='utf-8'))
        else:
            original=folder/'completion.json'
            measurements=json.loads(original.read_text(encoding='utf-8'))['recovery']
        for fold in ('confirmation','challenge'):
            for timestep,metrics in measurements[fold]['by_timestep'].items():
                rows.append({'pretrain_seed':int(seed),'arm':arm,'fold':fold,'timestep':int(timestep),
                             'nondefault_stereo_count':metrics['counts']['nondefault_stereo_count'],
                             'nondefault_stereo_correct':metrics['counts']['nondefault_stereo_correct'],
                             'nondefault_stereo_accuracy':metrics['nondefault_stereo_accuracy'],
                             'all_true_bond_stereo_accuracy':metrics['stereo_accuracy_true_bonds'],
                             'source':str(original.relative_to(ROOT)),
                             'source_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),
                             'checkpoint_sha256':item['sha256']})
result={'scope':'pretraining recovery heads, not a downstream stereochemical benchmark',
        'warning':'natural confirmation has only 3 non-default stereo bonds; challenge has 6 selected molecules; no stable conclusion',
        'no_stereo_based_selection_or_tuning':True,'rows':rows,'test_evaluated':False}
(run/'stereo_disclosure.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({'source_encoders':sum(len(arms) for arms in sources.values()),'diagnostic_rows':len(rows)}))
