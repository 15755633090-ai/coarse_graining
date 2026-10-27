"""Official-class vs current ASD numerical parity; training samples only."""
import ast
import copy
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem

ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
torch.set_num_threads(2)
official=ROOT/'research_code_audit/HMG-MRL'
sys.path[:0]=[str(official),str(official/'model')]
from config import cfg
from model.bipartite_transformer import GNN_atom_bond
from other_utils import initialize_weights

job=ROOT/'motif/runs/hmg_fusion_7arms_3seeds_20261025'
spec=importlib.util.spec_from_file_location('checked_current_hmg_ablation',job/'source/motif/hmg_fusion_ablation.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
reference=ROOT/'motif/runs/scale_pairs_3seeds_20261018/hmg_seed0'
manifest=json.loads((reference/'manifest.json').read_text(encoding='utf-8-sig'))
# Locally produced, previously SHA-256-verified input cache.
data=torch.load(ROOT/'motif/data/lipo.pt',map_location='cpu',weights_only=False)
selected=[r for r in data['molecules'] if r['split']=='train'][:2]
sys.path.insert(0,str(ROOT))
from motif.paper_data import load_hmg_features,hmg_inputs
features,assemble=load_hmg_features(ROOT/'motif/runs/hmg_features_finite_20261016/features.pickle',official)
kw,_,_=hmg_inputs(selected,features,assemble,'cuda:0')
cfg.device=torch.device('cuda:0');cfg.atom_dim_in=kw['x_atoms'].shape[-1];cfg.bond_dim_in=kw['x_bonds'].shape[-1]
initial=torch.load(reference/'initial_states.pt',map_location='cpu',weights_only=True)['full']
torch.manual_seed(2000)
default=GNN_atom_bond(cfg.atom_dim_in,cfg.bond_dim_in,1)
default_matches=all(torch.equal(v,initial[k]) for k,v in default.state_dict().items())
initialized=copy.deepcopy(default);initialize_weights(initialized)
different=[k for k,v in initialized.state_dict().items() if not torch.equal(v,initial[k])]
original=GNN_atom_bond(cfg.atom_dim_in,cfg.bond_dim_in,1);original.load_state_dict(initial,strict=True)
current=module.variant(original,'ASD');original.cuda();current.cuda()
target=torch.tensor([r['label'] for r in selected],device='cuda:0')
target=(target-manifest['target_mean'])/manifest['target_std']
def seed():
    random.seed(123);np.random.seed(123);torch.manual_seed(123);torch.cuda.manual_seed_all(123)
def compare(a,b):
    return float((a-b).abs().max())
original.eval();current.eval()
with torch.no_grad():
    eval_a=original(**kw);eval_b=current(**kw)
eval_error=compare(eval_a[1],eval_b[1])
def step(model):
    seed();model.train();model.zero_grad(set_to_none=True)
    output=model(**kw)
    mse=(output[1].flatten()-target).square().mean()
    loss=mse+.15*output[2];loss.backward()
    grads={k:(None if v.grad is None else v.grad.detach().clone()) for k,v in model.named_parameters()}
    torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=1e-5);optimizer.step()
    return output[1].detach(),float(loss.detach()),float(mse.detach()),float(output[2].detach()),grads
a=step(original);b=step(current)
gradient_errors=[];none_equal=True
for name,grad in a[4].items():
    other=b[4][name]
    none_equal &= (grad is None)==(other is None)
    if grad is not None and other is not None:gradient_errors.append(compare(grad,other))
updates=[compare(v,current.state_dict()[k]) for k,v in original.state_dict().items()]
report=dict(passed=eval_error<1e-6 and compare(a[0],b[0])<1e-6 and abs(a[1]-b[1])<1e-6
    and none_equal and max(gradient_errors)<1e-6 and max(updates)<1e-6,
    official_commit='6ad69261eb1af36c8c0b96594d79a7639b969223',training_rows=[r['row'] for r in selected],
    eval_output_max_error=eval_error,train_output_max_error=compare(a[0],b[0]),
    total_loss_error=abs(a[1]-b[1]),mse_error=abs(a[2]-b[2]),auxiliary_error=abs(a[3]-b[3]),
    gradients_none_pattern_equal=none_equal,gradient_max_error=max(gradient_errors),
    one_adamw_step_parameter_max_error=max(updates),
    archived_initialization_exactly_matches_default_constructor_seed_2000=default_matches,
    tensors_different_after_official_initializer=len(different),
    first_different_initializer_keys=different[:15],
    scope='Aligned local regression objective and AdamW for parity, not evidence that AdamW or initialization matches the paper',
    test_forward=False)
(OUT/'full_numerical_parity.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False))
