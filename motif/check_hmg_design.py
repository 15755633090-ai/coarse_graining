"""Real training-input path interventions and validation-only design checks."""
import copy
import importlib
import json
from pathlib import Path
import torch
from .data import save_json
from .paper_data import load_paper,load_hmg_features,hmg_inputs
from .hmg_protocol_v2 import DescriptorScaler,apply_regression_settings,digest
from .hmg_design_models import IsolatedMother,make_model,objective,capacity_audit,match_capacity
from .hmg_study_controls import standardized_output_initialization,convergence_record,budget_decision,complete_design_plan,seal_panel

def main():
    torch.set_num_threads(2)
    prepared=Path('motif/runs/hmg_protocol_v2_prepared_fix1_20261026')
    out=Path('motif/runs/hmg_design_v3_checks_20261026');out.mkdir(exist_ok=False)
    data=torch.load(prepared/'raw_descriptors.pt',weights_only=True)
    split=json.loads((prepared/'seed_2020_split.json').read_text())
    positions=split['positions']['train'][:4];rows=[data['molecules'][i] for i in positions]
    paper=Path('research_code_audit/HMG-MRL').resolve()
    cls,_,cfg=load_paper(paper,'HMG-MRL','cuda:0');apply_regression_settings(cfg)
    features,assemble=load_hmg_features(Path('motif/runs/hmg_features_finite_20261016/features.pickle'),paper)
    inputs=hmg_inputs(rows,features,assemble,'cuda:0')[0]
    scaler=DescriptorScaler.from_state_dict(torch.load(prepared/'seed_2020_descriptor_scaler.pt',weights_only=True))
    inputs['molecule_descriptor']=torch.tensor(scaler.transform(data['raw'].numpy())[positions],device='cuda:0')
    cfg.atom_dim_in=inputs['x_atoms'].shape[-1];cfg.bond_dim_in=inputs['x_bonds'].shape[-1]
    utils=importlib.import_module('other_utils');utils.seed_everything(2020)
    full=cls(cfg.atom_dim_in,cfg.bond_dim_in,1);utils.initialize_weights(full)
    mother=IsolatedMother(full);target=torch.tensor([r['label'] for r in rows],device='cuda:0')
    checks={}
    reference=copy.deepcopy(full).cuda().eval()
    native=make_model(full,'ASD').cuda().eval()
    torch.testing.assert_close(reference(**inputs)[1],native(**inputs)[1],rtol=0,atol=0)
    checks['native_full_exact']=True
    del reference,native
    for arm in ('A','S','D'):
        model=make_model(full,arm,'isolated',isolated_mother=mother).cuda().eval()
        changed={k:v.clone() for k,v in inputs.items()}
        if arm!='D':changed['molecule_descriptor']+=3
        if arm!='S':changed['frag_src_features']+=2
        else:
            changed['x_atoms']+=2;changed['x_bonds']+=2
            changed['frag_src_features'][:,0]+=5
        with torch.no_grad():torch.testing.assert_close(model(**inputs)[1],model(**changed)[1],rtol=0,atol=0)
        # D must also ignore fragment and atom inputs.
        if arm=='D':
            changed['x_atoms']+=7
            with torch.no_grad():torch.testing.assert_close(model(**inputs)[1],model(**changed)[1],rtol=0,atol=0)
        checks['isolated_'+arm+'_intervention']=True
        del model
    for study,arm in [('native','A'),('noaux','A'),('single-direct','S'),('single-direct-noaux','D'),('common-head','AS'),('common-head-noaux','SD')]:
        model=make_model(full,arm,study).cuda().eval();alpha=0 if 'noaux' in study else .15
        audit=capacity_audit(model,inputs,target,alpha)
        loss,_,_=objective(model(**inputs),target,alpha=alpha);loss.backward()
        assert torch.isfinite(loss)
        checks[study]=audit;del model
    for study in ('common-head','isolated'):
        alpha=0 if study=='isolated' else .15
        ref=make_model(full,'ASD',study,isolated_mother=mother).cuda()
        budget=capacity_audit(ref,inputs,target,alpha)['objective_connected_parameters'];del ref
        model=make_model(full,'A',study,isolated_mother=mother).cuda()
        checks[study+'_capacity']=match_capacity(model,budget,inputs,target,alpha)
        model.eval()
        with torch.no_grad():before=model(**inputs)[1].clone()
        standardized_output_initialization(model,2.3,1.7)
        with torch.no_grad():torch.testing.assert_close(model(**inputs)[1]*1.7+2.3,before,rtol=1e-4,atol=1e-4)
        del model
    history=[dict(epoch=1,valid_rmse=1.),dict(epoch=2,valid_rmse=.9)]
    record=convergence_record(history,2)
    assert budget_decision([record],[2,4])['action']=='extend_panel_cap'
    plan=complete_design_plan(digest(prepared/'manifest.json'),[2020,2021,2022,2023,2024])
    assert len(plan['cases'])==340
    save_json(Path('motif/runs/hmg_design_v3_plan_20261026.json'),plan)
    try:seal_panel(plan,[])
    except ValueError:checks['missing_panel_rejected']=True
    else:raise AssertionError('Missing cases accepted')
    save_json(out/'acceptance.json',dict(passed=True,checks=checks,test_evaluated=False,formal_training_resumed=False))
    print('Design checks passed; no test evaluation; formal training remains paused.',flush=True)

if __name__=='__main__':main()
