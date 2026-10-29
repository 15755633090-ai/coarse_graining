"""Predeclared likelihood-skip and observation controls; no test labels."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
import torch
from _github_sync import search_adjacency_rescue_20261029 as base
from motif.adjacency_diffusion.posterior import PosteriorModel,likelihood_log_odds


def prepare(args):
    out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=False)
    old=json.loads((ROOT/'motif/adjacency_diffusion/runs/rescue_screen_20261029/plan.json').read_text(encoding='utf-8'))
    plan=dict(old)
    corpus=ROOT/'motif/brics_atom_diffusion/runs/pilot_zinc4096_esol3_20261029/corpus'
    plan.update(train_smiles=pd.read_csv(corpus/'train.csv').smiles.tolist()[:256],
                development_smiles=pd.read_csv(corpus/'valid.csv').smiles.tolist()[256:320],
                confirmation_smiles=pd.read_csv(corpus/'valid.csv').smiles.tolist()[320:384],
                steps=args.steps,experiment='Gaussian likelihood skip and observation control',
                seeds=[7,1],selection_rule='none; report all four configurations on both folds',
                confirmation_rule='no selection or retuning; all cells evaluated',
                row_ranges={'train':[0,256],'development':[256,320],'confirmation':[320,384]},
                prior_corpus_used_before=True,device=args.device)
    common=dict(lr=.001,prior=True,gate=True,initialization='density',balanced=False,
                degree_weight=0.,mode='bounded_A',policy='random')
    plan['variants']=[dict(common,id='bounded_reference',posterior=False,evidence=True),
                      dict(common,id='likelihood_skip',posterior=True,evidence=True),
                      dict(common,id='likelihood_skip_no_brics',posterior=True,evidence=True,prior=False,gate=False),
                      dict(common,id='structure_only_control',posterior=True,evidence=False)]
    files=[ROOT/name for name in old['source_sha256'] if (ROOT/name).exists()]
    files += [ROOT/'motif/adjacency_diffusion/posterior.py',ROOT/'motif/adjacency_diffusion/test_posterior.py',Path(__file__).resolve()]
    files=sorted(set(files))
    for path in files:
        dest=out/'frozen_source'/path.relative_to(ROOT)
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,dest)
    plan['source_sha256']={p.relative_to(ROOT).as_posix():base.sha(p) for p in files}
    base.atomic_write(out/'plan.json',plan)
    print(json.dumps({'prepared':str(out),'cells':8,'steps':args.steps}))


def execute(args):
    out=args.output_dir.resolve()
    plan=json.loads((out/'plan.json').read_text(encoding='utf-8'))
    if ROOT!=out/'frozen_source':
        raise ValueError('Frozen source required')
    for name,digest in plan['source_sha256'].items():
        if base.sha(ROOT/name)!=digest:
            raise ValueError('Changed source '+name)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if plan['device'].startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(.4,0)
    train=base.GraphDataset(plan['train_smiles'])
    dev=base.GraphDataset(plan['development_smiles'])
    confirm=base.GraphDataset(plan['confirmation_smiles'])
    density=base.positive_rate(train)
    original=base.make_model
    def factory(spec,plan,seed,density):
        reference=original(spec,plan,seed,density)
        if not spec['posterior']:
            return reference
        model=PosteriorModel(reference.config,spec['evidence']).to(plan['device'])
        model.load_state_dict(reference.state_dict(),strict=True)
        model.set_stage('pretrain')
        return model
    base.make_model=factory
    rows=[]
    started=time.perf_counter()
    for seed in plan['seeds']:
        for spec in plan['variants']:
            model,row=base.train_cell(spec,seed,'screen',out,plan,train,dev,density,len(rows))
            row['confirmation']=base.evaluate(model,confirm,spec,plan,density)
            # The generic helper's intermediate checkpoint is never exposed as a usable epsilon checkpoint.
            path=out/'screen'/(spec['id']+'_seed_'+str(seed))/'encoder_experiment.pt'
            checkpoint=torch.load(path,map_location='cpu',weights_only=True)
            checkpoint['experimental_schema']='gaussian_likelihood_skip_v1'
            checkpoint['head_semantics']='adjacency logits; optional Gaussian likelihood skip; NOT epsilon'
            base.atomic_write(path,checkpoint,binary=True)
            rows.append(row)
            base.atomic_write(out/'results.json',{'runs':rows,'test_evaluated':False,'selection':None})
            lines=['# Likelihood skip diagnostic: all cells reported','',
                   '| variant | seed | dev ratio t200/500 | confirm ratio t200/500 | confirm AP500 | confirm AP1000 |',
                   '|---|---:|---:|---:|---:|---:|']
            for r in rows:
                c=r['confirmation']
                lines.append(f'| {r["variant"]["id"]} | {r["seed"]} | {r["development_mid_score"]:.4f} | '
                             f'{base.mid_score(c):.4f} | {c["500"]["average_precision"]:.4f} | {c["1000"]["average_precision"]:.4f} |')
            (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    base.atomic_write(out/'status.json',{'state':'complete','completed_cells':len(rows),
                      'elapsed_seconds':time.perf_counter()-started,'test_evaluated':False})


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--steps',type=int,default=750)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    if args.execute:
        try:
            execute(args)
        except Exception as exc:
            base.atomic_write(args.output_dir/'status.json',{'state':'failed','error':repr(exc)})
            raise
    else:
        prepare(args)


if __name__=='__main__':
    main()

