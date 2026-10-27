"""Frozen-source native seven-arm queue; paired budget extension then sealed test."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from .data import save_json
from .hmg_protocol_v2 import digest
from .hmg_study_controls import complete_design_plan,budget_decision

def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();root=a.workspace;run=a.run
    prepared=root/'motif/runs/hmg_protocol_v2_prepared_fix1_20261026'
    baseline=run/'baseline';fusion=run/'fusion';ladder=[100,200,400]
    plan=complete_design_plan(digest(prepared/'manifest.json'),[2020,2021,2022,2023,2024],ladder)
    plan.update(cases=[c for c in plan['cases'] if c['study']=='native'],target_scaling='raw',scheduler='constant',
                formal_training_paused=False,metadata_only=False,accept_budget_limited=True,
                budget_limited_policy='predeclared maximum 400; report budget-limited, not convergence',
                local_choices_not_author_verified=['exact scaffold rows','target scaling','constant scheduler','budget ladder'])
    save_json(run/'plan.json',plan)
    def command(module,args):
        save_json(run/'queue_status.json',dict(phase='running',module=module,args=[str(v) for v in args]))
        print('START',module,*args,flush=True)
        subprocess.run([sys.executable,'-u','-m',module,*map(str,args)],check=True)
    shared=['--prepared',prepared,'--paper-root',root/'research_code_audit/HMG-MRL',
        '--official-tree',root/'motif/audits/hmg_reliability_20261026/official_tree.json',
        '--feature-cache',root/'motif/runs/hmg_features_finite_20261016/features.pickle',
        '--study','native','--max-epochs',400,'--target-scaling','raw','--scheduler','constant',
        '--budget-ladder',100,200,400]
    for cap in ladder:
        for stage,folder in [('baseline',baseline),('fusion',fusion)]:
            args=[*shared,'--stage',stage,'--output',folder,'--active-cap',cap]
            if folder.exists():args+=['--resume']
            if stage=='fusion':args+=['--baseline',baseline]
            command('motif.train_hmg_design',args)
        artifacts=read(fusion/'results_partial.json')
        decisions={str(seed):budget_decision([r['convergence'] for r in artifacts if r['seed']==seed],ladder)
                   for seed in [2020,2021,2022,2023,2024]}
        save_json(run/f'budget_decision_{cap}.json',decisions)
        if all(v['action']=='validation_complete' for v in decisions.values()):break
    command('motif.seal_hmg_design',['--plan',run/'plan.json','--runs',fusion,'--output',run/'seal.json'])
    command('motif.evaluate_hmg_sealed',['--seal',run/'seal.json','--prepared',prepared,
        '--paper-root',root/'research_code_audit/HMG-MRL','--feature-cache',root/'motif/runs/hmg_features_finite_20261016/features.pickle',
        '--output',run/'evaluation'])
    save_json(run/'queue_status.json',dict(phase='complete',test_evaluated=True,author_exact_protocol=False))

if __name__=='__main__':
    try:main()
    except Exception as error:
        print('QUEUE FAILED:',repr(error),flush=True)
        raise
