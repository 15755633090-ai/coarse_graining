"""Replay smoke VALIDATION predictions through the sealed evaluator loader."""
import csv
from pathlib import Path
import torch
from .data import save_json
from .train_hmg_design import read
from .evaluate_hmg_sealed import restore_model
from .paper_data import load_paper,load_hmg_features,hmg_inputs
from .hmg_protocol_v2 import DescriptorScaler,apply_regression_settings

def main():
    torch.set_num_threads(2);prepared=Path('motif/runs/hmg_protocol_v2_prepared_fix1_20261026')
    data=torch.load(prepared/'raw_descriptors.pt',weights_only=True)
    split=read(prepared/'seed_2020_split.json');positions=split['positions']['valid'][:16]
    rows=[data['molecules'][i] for i in positions]
    paper=Path('research_code_audit/HMG-MRL').resolve()
    cls,_,cfg=load_paper(paper,'HMG-MRL','cuda:0');apply_regression_settings(cfg)
    features,assemble=load_hmg_features(Path('motif/runs/hmg_features_finite_20261016/features.pickle'),paper)
    inputs=hmg_inputs(rows,features,assemble,'cuda:0')[0]
    cfg.atom_dim_in=inputs['x_atoms'].shape[-1];cfg.bond_dim_in=inputs['x_bonds'].shape[-1]
    scaler=DescriptorScaler().fit(data['raw'].numpy(),split['positions']['train'])
    inputs['molecule_descriptor']=torch.tensor(scaler.transform(data['raw'].numpy())[positions],device='cuda:0')
    checks=[]
    for study in ('native','isolated'):
        run=Path(f'motif/runs/hmg_design_v3_{study}_smoke_20261026');protocol=read(run/'protocol.json')
        for result in read(run/'results_partial.json'):
            folder=run/'seed_2020'/result['arm'];saved=torch.load(folder/'best.pt',weights_only=True)
            model=restore_model(cls,cfg,protocol,saved,folder,'cuda:0')
            with torch.no_grad():pred=(model(**inputs)[1].flatten()*saved['std']+saved['mean']).cpu()
            with (folder/'valid_predictions.csv').open(encoding='utf-8-sig',newline='') as stream:
                expected=torch.tensor([float(r['prediction']) for r in csv.DictReader(stream)])
            torch.testing.assert_close(pred,expected,rtol=0,atol=0)
            checks.append(f"{study}/{result['arm']}");del model
    save_json(Path('motif/runs/hmg_design_v3_evaluation_restore_20261026.json'),
        dict(passed=True,exact_validation_replays=checks,test_evaluated=False,formal_training_resumed=False))
    print('14 exact validation replays passed; no test predictions made.')

if __name__=='__main__':main()
