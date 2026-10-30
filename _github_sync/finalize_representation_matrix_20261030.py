"""Verify every cell; report objective/data effects without compute-equivalence claims."""
import hashlib,json,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from motif.adjacency_diffusion.masked_bonds import MaskedBondModel,load_model as load_edge
from motif.adjacency_diffusion.representation_pretrain import load_model as load_repr
from motif.adjacency_diffusion.model import Config
from _github_sync.search_adjacency_rescue_20261029 import named_initialize
from _github_sync.check_masked_property_20261029 import tensor_hash
out=ROOT/'motif/adjacency_diffusion/runs/representation_matrix_20261030'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
status=read(out/'status.json');assert status['state']=='complete'
plan=read(out/'plan.json');sources=read(out/'sources.json');result=read(out/'results.json')
assert len(sources)==12 and len(result['runs'])==12 and status['new_property_runs']==9
for p,h in plan['source_sha256'].items():assert hashlib.sha256((out/'frozen_source'/p).read_bytes()).hexdigest()==h,p
old=Path(plan['old_run']);original=read(old/'results.json')
for corpus in ['small','expanded']:
    for seed in [7,1,2]:
        edge=sources[f'{corpus}_edge_{seed}'];rep=sources[f'{corpus}_representation_{seed}']
        assert edge['initial_state_hash']==rep['initial_state_hash']
        for obj,item in [('edge',edge),('representation',rep)]:
            cp=Path(item['path']);assert hashlib.sha256(cp.read_bytes()).hexdigest()==item['checkpoint_sha256']
            model,saved=(load_repr if obj=='representation' else load_edge)(cp)
            model.set_stage('pretrain')
            item['active_pretraining_parameters_posthoc']=sum(p.numel() for p in model.parameters() if p.requires_grad)
            if not item.get('reused_original'):
                assert item['checkpoint_replay_passed'] and len(read(cp.parent/'history.json'))==12
                assert item['graph_draws']==9600 and item['encoder_graph_passes']==9600*(2 if obj=='representation' else 1)
            else:
                prior=read(old/'pretraining'/f'masked_seed_{seed}'/'completion.json')
                assert prior['checkpoint_replay_passed']
                item['masked_candidate_positions']=prior['supervised_candidate_pairs_accumulated']
                item['masked_true_bonds']=prior['supervised_true_bonds_accumulated']
        assert edge['masked_candidate_positions']==rep['masked_candidate_positions']
        assert edge['masked_true_bonds']==rep['masked_true_bonds']
for row in result['runs']:
    if row.get('reused_original'):continue
    key=f"{row['corpus']}_{row['objective']}_{row['seed']}";folder=out/'properties'/key
    h=read(folder/'history.json');cfg=read(folder/'config.json')
    assert len(h)==60 and row['checkpoint_replay_passed']
    assert min(v['validation']['rmse'] for v in h)==row['validation']['rmse']
    prior=read(old/f"scratch_task_seed_{row['task_seed']}"/'config.json')
    assert cfg['splits']==prior['splits'] and cfg['head_initial_hash']==prior['head_initial_hash']
    new,_=load_edge(folder/'best.pt');new.set_stage('finetune')
    ref,_=load_edge(old/f"scratch_task_seed_{row['task_seed']}"/'best.pt');ref.set_stage('finetune')
    assert sum(p.numel() for p in new.parameters() if p.requires_grad)==sum(p.numel() for p in ref.parameters() if p.requires_grad)
    # Corpus-derived buffers can differ, but all trainable initial encoder weights match.
    new_cfg=Config(**new.config.__dict__)
    new_init=MaskedBondModel(new_cfg,{name:getattr(new,'prior_'+name) for name in ['type','stereo','conjugated','ring']},.3)
    ref_init=MaskedBondModel(new_cfg,{name:getattr(ref,'prior_'+name) for name in ['type','stereo','conjugated','ring']},.3)
    named_initialize(new_init,row['task_seed']);new_init.initialize_priors()
    named_initialize(ref_init,row['task_seed']);ref_init.initialize_priors()
    assert tensor_hash(dict(new_init.encoder.named_parameters()))==tensor_hash(dict(ref_init.encoder.named_parameters()))
effects=[]
for seed in [7,1,2]:
    r={(x['corpus'],x['objective']):x['validation']['rmse'] for x in result['runs'] if x['seed']==seed}
    effects.append({'seed':seed,'representation_minus_edge_small':r['small','representation']-r['small','edge'],
        'representation_minus_edge_expanded':r['expanded','representation']-r['expanded','edge'],
        'expanded_minus_small_edge':r['expanded','edge']-r['small','edge'],
        'expanded_minus_small_representation':r['expanded','representation']-r['small','representation'],
        'interaction':(r['expanded','representation']-r['expanded','edge'])-(r['small','representation']-r['small','edge'])})
result['paired_factor_effects']=effects;write(out/'results.json',result);write(out/'sources.json',sources)
write(out/'acceptance.json',{'state':'complete','source_hashes_match':True,'all_checkpoint_replays_passed':True,
    'initializations_paired':True,'all_new_tasks_60_epochs':True,'identical_downstream_architecture_capacity':True,
    'matched_corruption_positions_across_objectives':True,'strict_compute_match':False,'test_evaluated':False})
lines=['# Objective x corpus representation study','','| corpus | objective | validation RMSE mean ± sample SD |','|---|---|---:|']
for key,m in result['aggregate'].items():
    corpus,obj=key.split('_',1);lines.append(f"| {corpus} | {obj} | {m['mean_rmse']:.5f} ± {m['sample_sd']:.5f} |")
s=result['scratch_reference'];lines+=['',f"Original scratch reference: {s['mean_rmse']:.5f} ± {s['sample_sd']:.5f}.",'',
    '| pretrain seed | objective Δsmall | objective Δexpanded | data Δedge | data Δrepresentation |','|---:|---:|---:|---:|---:|']
for e in effects:lines.append(f"| {e['seed']} | {e['representation_minus_edge_small']:+.5f} | {e['representation_minus_edge_expanded']:+.5f} | {e['expanded_minus_small_edge']:+.5f} | {e['expanded_minus_small_representation']:+.5f} |")
lines+=['','Negative differences are better. Paired graph draws/updates, not strict compute parity. Same-source ZINC expansion is not domain alignment. All validation splits have been reused; no test evaluation.','',
    '| source | clean effective rank | clean dimension std | unique molecules seen | encoder graph passes |','|---|---:|---:|---:|---:|']
for key,item in sources.items():
    d=item['diagnostics']['clean'];lines.append(f"| {key} | {d['effective_rank']:.4f} | {d['mean_dimension_std']:.4f} | {item.get('unique_molecules_seen','original')} | {item.get('encoder_graph_passes','9600, original')} |")
lines+=['','Low rank is not constant collapse and does not by itself prove task uselessness. The small-batch graph adaptation is not the vision LeJEPA recipe. Untrained chemical recovery heads are not evaluated as representation recovery scores.']
(out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'aggregate':result['aggregate'],'effects':effects,'acceptance':'passed'}))
