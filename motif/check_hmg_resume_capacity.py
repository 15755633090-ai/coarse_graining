"""Training-only checks: all capacity controls and real HMG Adam replay."""
import copy
import importlib
from pathlib import Path
import torch
from .data import save_json
from .train_hmg_design import read,atomic_save
from .paper_data import load_paper,load_hmg_features,hmg_inputs
from .hmg_protocol_v2 import DescriptorScaler,apply_regression_settings,build_optimizer
from .hmg_design_models import IsolatedMother,make_model,capacity_audit,match_capacity,objective,auxiliary_weight

def main():
    torch.set_num_threads(2)
    out=Path('motif/runs/hmg_design_v3_resume_capacity_20261027');out.mkdir(exist_ok=False)
    prepared=Path('motif/runs/hmg_protocol_v2_prepared_fix1_20261026')
    data=torch.load(prepared/'raw_descriptors.pt',weights_only=True)
    positions=read(prepared/'seed_2020_split.json')['positions']['train'][:4]
    rows=[data['molecules'][i] for i in positions]
    paper=Path('research_code_audit/HMG-MRL').resolve()
    cls,_,cfg=load_paper(paper,'HMG-MRL','cuda:0');apply_regression_settings(cfg)
    features,assemble=load_hmg_features(Path('motif/runs/hmg_features_finite_20261016/features.pickle'),paper)
    inputs=hmg_inputs(rows,features,assemble,'cuda:0')[0]
    scaler=DescriptorScaler.from_state_dict(torch.load(prepared/'seed_2020_descriptor_scaler.pt',weights_only=True))
    inputs['molecule_descriptor']=torch.tensor(scaler.transform(data['raw'].numpy())[positions],device='cuda:0')
    target=torch.tensor([r['label'] for r in rows],device='cuda:0')
    cfg.atom_dim_in=inputs['x_atoms'].shape[-1];cfg.bond_dim_in=inputs['x_bonds'].shape[-1]
    utils=importlib.import_module('other_utils');utils.seed_everything(2020)
    full=cls(cfg.atom_dim_in,cfg.bond_dim_in,1);utils.initialize_weights(full);mother=IsolatedMother(full)
    matches=[]
    for study in ('common-head','common-head-noaux','isolated'):
        alpha=auxiliary_weight(study)
        for combination in ('AS','AD','SD','ASD'):
            reference=make_model(full,combination,study,isolated_mother=mother).cuda()
            budget=capacity_audit(reference,inputs,target,alpha)['objective_connected_parameters'];del reference
            for arm in combination:
                model=make_model(full,arm,study,isolated_mother=mother).cuda()
                result=match_capacity(model,budget,inputs,target,alpha)
                matches.append(dict(study=study,arm=arm,combination=combination,width=result['width'],
                    relative_error=result['relative_error'],actual=result['actual'],target=budget))
                del model
        print('Capacity controls passed: '+study,flush=True)
    assert len(matches)==27
    replay=[]
    def step(model,optimizer,epoch,alpha):
        utils.seed_everything(2020+4000+epoch);model.train()
        loss=objective(model(**inputs),target,alpha=alpha)[0]
        assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True)
        optimizer.step()
        return float(loss.detach())
    for study in ('native','noaux','common-head-noaux','single-direct','single-direct-noaux','isolated'):
        alpha=auxiliary_weight(study)
        model=make_model(full,'A',study,isolated_mother=mother).cuda();optimizer=build_optimizer(model)
        step(model,optimizer,1,alpha)
        path=out/(study+'.pt')
        atomic_save(dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                         optimizer=optimizer.state_dict()),path)
        expected_loss=step(model,optimizer,2,alpha)
        expected={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        restored=make_model(full,'A',study,isolated_mother=mother).cuda();opt=build_optimizer(restored)
        saved=torch.load(path,weights_only=True,map_location='cpu')
        restored.load_state_dict(saved['model'],strict=True);opt.load_state_dict(saved['optimizer'])
        actual_loss=step(restored,opt,2,alpha)
        assert actual_loss==expected_loss
        for name,value in restored.state_dict().items():torch.testing.assert_close(value.cpu(),expected[name],rtol=0,atol=0)
        # Momentum and step counters, not just output weights, must replay exactly.
        expected_opt=optimizer.state_dict();actual_opt=opt.state_dict()
        assert expected_opt['param_groups']==actual_opt['param_groups']
        for key,state in expected_opt['state'].items():
            for name,value in state.items():torch.testing.assert_close(value,actual_opt['state'][key][name],rtol=0,atol=0)
        replay.append(dict(study=study,loss_exact=True,weights_exact=True,optimizer_exact=True))
        del model,optimizer,restored,opt;torch.cuda.empty_cache()
        print('Epoch-boundary replay passed: '+study,flush=True)
    save_json(out/'acceptance.json',dict(passed=True,capacity_matches=matches,epoch_boundary_replays=replay,
        rows=positions,test_evaluated=False,formal_training_resumed=False,scope='four training molecules; two update replay, not performance'))

if __name__=='__main__':main()
