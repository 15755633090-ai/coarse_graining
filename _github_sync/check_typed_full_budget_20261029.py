"""Fork original latest states, disable early stopping, finish the same 60-epoch cap."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
EXPERIMENT=ROOT/'motif/adjacency_diffusion/runs/typed_property_comparison_20261029'
sys.path.insert(0,str(EXPERIMENT/'frozen_source'))
import torch
from motif.adjacency_diffusion import typed_train as runner
from motif.adjacency_diffusion.typed_refined import load_model
from motif.adjacency_diffusion.typed_bonds import GraphDataset


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out=ROOT/'motif/adjacency_diffusion/runs/typed_full_budget_diagnostics_20261029'
    out.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_per_process_memory_fraction(.45,0)
    original=json.loads((EXPERIMENT/'plan.json').read_text(encoding='utf-8'))
    inputs,full_splits,spec=runner.io.build_property_data(Path(original['data_root']),'esol')
    splits={k:v for k,v in full_splits.items() if k in ('train','valid')}
    dataset=GraphDataset(inputs.smiles,inputs.targets,inputs.target_mask)
    scaler=runner.io.TargetScaler.fit(inputs.targets[splits['train']],inputs.target_mask[splits['train']])
    results=json.loads((EXPERIMENT/'results.json').read_text(encoding='utf-8'))['runs']
    plan={'change':'patience=1000; retain the original 60-epoch cap; exact optimizer/model/RNG/loader state continuation',
          'frozen_code':str(EXPERIMENT/'frozen_source'),'original_plan_sha256':sha(EXPERIMENT/'plan.json'),
          'script_sha256':sha(Path(__file__)),'test_evaluated':False,
          'source_latest_sha256':{f'{r["arm"]}_{r["task_seed"]}':sha(EXPERIMENT/f'{r["arm"]}_task_seed_{r["task_seed"]}'/'latest.pt') for r in results}}
    runner.io.atomic_write(out/'plan.json',plan)
    rows=[]
    for row in results:
        seed,arm=row['task_seed'],row['arm']
        source=EXPERIMENT/f'{arm}_task_seed_{seed}'
        if row['epochs_run']>=60:
            result={**row,'continued':False,'original_best_rmse':row['validation']['rmse']}
        else:
            folder=out/f'{arm}_task_seed_{seed}'
            folder.mkdir()
            model,saved=load_model(source/'latest.pt')
            model.to('cuda:0')
            metadata=json.loads((source/'config.json').read_text(encoding='utf-8'))
            metadata.update(continuation_source=str(source),continuation_latest_sha256=sha(source/'latest.pt'),
                            continuation_patience=1000,continuation_plan_sha256=sha(out/'plan.json'))
            runner.io.atomic_write(folder/'config.json',metadata)
            # Only bind the copied checkpoint's metadata to its new audited directory.
            # Parameter, optimizer, epoch/history, sampler and RNG fields remain untouched.
            saved['run_metadata']=metadata
            saved['run_config_sha256']=sha(folder/'config.json')
            saved['best_checkpoint']=copy.deepcopy(saved['best_checkpoint'])
            saved['best_checkpoint']['run_metadata']=metadata
            saved['best_checkpoint']['run_config_sha256']=saved['run_config_sha256']
            runner.io.atomic_write(folder/'latest.pt',saved,binary=True)
            settings=SimpleNamespace(stage='finetune',output_dir=folder,device='cuda:0',seed=seed,
                epochs=60,patience=1000,batch_size=16,learning_rate=.001,weight_decay=0.,grad_clip=5.,resume=True)
            train=runner.io.loader(dataset,splits['train'],settings,True)
            valid=runner.io.loader(dataset,splits['valid'],settings,False)
            runner.io.atomic_write(out/'status.json',{'state':'running','arm':arm,'seed':seed,'completed_cases':len(rows),'planned_cases':9})
            result=runner.fit(model,train,valid,settings,metadata,spec,scaler,None)
            pred,target,mask,_=runner.io.predict(model,valid,'cuda:0',scaler,spec)
            result.update(arm=arm,task_seed=seed,validation=runner.io.compute_property_metrics(pred,target,mask,spec),
                          continued=True,original_best_rmse=row['validation']['rmse'],test_evaluated=False)
            replay,_=load_model(folder/'best.pt')
            replay.to('cuda:0')
            actual,*_=runner.io.predict(replay,valid,'cuda:0',scaler,spec)
            torch.testing.assert_close(actual,pred,atol=1e-6,rtol=1e-6)
            result['checkpoint_replay_passed']=True
            runner.io.atomic_write(folder/'completion.json',result)
            del model,replay
        rows.append(result)
        runner.io.atomic_write(out/'results.json',{'runs':rows,'test_evaluated':False})
        print(arm,seed,'original',row['validation']['rmse'],'full cap',result['validation']['rmse'],flush=True)
    runner.io.atomic_write(out/'status.json',{'state':'complete','completed_cases':9,'continued_cases':7,'test_evaluated':False})


if __name__=='__main__':
    main()
