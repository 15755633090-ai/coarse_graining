"""Freeze validation-selected checkpoints; never reads test data or scores."""
import argparse
import json
from pathlib import Path
from .data import save_json
from .hmg_protocol_v2 import digest
from .hmg_study_controls import seal_panel
from .hmg_study_controls import convergence_record
import hashlib

def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();artifacts=[]
    if a.output.exists():raise ValueError('Seal already exists; refusing overwrite')
    for run in a.runs:
        protocol=read(run/'protocol.json')
        if protocol.get('smoke_only'):raise ValueError('Smoke cannot enter formal comparison')
        for item in read(run/'results_partial.json'):
            name=item['case_id'].split('/',1)[1]
            checkpoint=run/f"seed_{item['seed']}"/name/'best.pt'
            if digest(checkpoint)!=item['checkpoint_sha256']:raise ValueError('Selected checkpoint changed')
            history=read(checkpoint.parent/'history.json')
            best=min(history,key=lambda r:r['valid_rmse'])
            if best['epoch']!=item['best_epoch']:raise ValueError('Checkpoint not validation-selected')
            if convergence_record(history,item['convergence']['budget'])!=item['convergence']:
                raise ValueError('Stopping metadata differs from validation history')
            item=dict(item,run_directory=str(run.resolve()),protocol_sha256=digest(run/'protocol.json'))
            artifacts.append(item)
    seal=seal_panel(read(a.plan),artifacts)
    lookup={(v['seed'],v['case_id']):v for v in artifacts}
    for case in seal['cases']:
        source=lookup[case['seed'],case['case_id']]
        case.update(run_directory=source['run_directory'],protocol_sha256=source['protocol_sha256'])
    del seal['seal_sha256']
    seal['seal_sha256']=hashlib.sha256(json.dumps(seal,sort_keys=True).encode()).hexdigest()
    save_json(a.output,seal)
    print('Validation panel sealed. No test data were accessed.')

if __name__=='__main__':main()
