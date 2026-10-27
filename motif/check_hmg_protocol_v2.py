"""Leakage guards, raw schema identity, optimizer, and epoch-boundary replay."""
import argparse
import copy
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import save_json
from .hmg_protocol_v2 import VERSION,DescriptorScaler,build_optimizer,require_v2_checkpoint


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2);a.output.mkdir(parents=True,exist_ok=False)
    raw=np.array([[1.,np.nan,7.],[2.,4.,7.],[3.,8.,7.],[1e8,-2e8,9.],[np.inf,100.,7.]])
    first=DescriptorScaler().fit(raw,[0,1,2]);changed=raw.copy();changed[3:]=9e15
    second=DescriptorScaler().fit(changed,[0,1,2])
    for name in ('mean','scale','median','constant'):assert np.array_equal(getattr(first,name),getattr(second,name))
    train=first.transform(raw)[[0,1,2]]
    np.testing.assert_allclose(train[:,:2].mean(0),0,atol=1e-6)
    np.testing.assert_allclose(train[:,:2].std(0),1,atol=1e-6)
    assert np.isfinite(first.transform(raw)).all() and (train[:,2]==0).all()
    torch.save(first.state_dict(),a.output/'scaler.pt')
    restored=DescriptorScaler.from_state_dict(torch.load(a.output/'scaler.pt',weights_only=True))
    assert np.array_equal(first.transform(raw),restored.transform(raw))
    try:require_v2_checkpoint({'model':{}},'test',2020)
    except ValueError:pass
    else:raise AssertionError('Legacy checkpoint accepted')
    data=torch.load(a.prepared/'raw_descriptors.pt',weights_only=True)
    assert len(data['names'])==len(set(data['names']))==200
    # A raw physical descriptor must not still be a [0,1] CDF value.
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    mol=Chem.MolFromSmiles(data['molecules'][0]['smiles'])
    mw=data['names'].index('MolWt')
    assert abs(float(data['raw'][0,mw])-Descriptors.MolWt(mol))<1e-8
    for seed in (2020,2021,2022,2023,2024):
        import json
        split=json.loads((a.prepared/f'seed_{seed}_split.json').read_text(encoding='utf-8'))
        state=torch.load(a.prepared/f'seed_{seed}_descriptor_scaler.pt',weights_only=True)
        assert state['fit_positions']==split['positions']['train'] and split['scaffold_overlap']==0
    torch.manual_seed(1)
    template=nn.Sequential(nn.Linear(3,8),nn.ReLU(),nn.Dropout(.2),nn.Linear(8,1))
    uninterrupted=copy.deepcopy(template);prefix=copy.deepcopy(template)
    opt1=build_optimizer(uninterrupted);opt2=build_optimizer(prefix)
    assert type(opt1) is torch.optim.Adam and opt1.param_groups[0]['lr']==.001
    x=torch.randn(12,3);y=torch.randn(12,1)
    def epoch(model,opt,n):
        torch.manual_seed(100+n);model.train()
        for batch in (slice(0,6),slice(6,12)):
            loss=(model(x[batch])-y[batch]).square().mean();opt.zero_grad();loss.backward();opt.step()
    for n in range(1,4):epoch(uninterrupted,opt1,n)
    for n in range(1,3):epoch(prefix,opt2,n)
    torch.save(dict(model=prefix.state_dict(),optimizer=opt2.state_dict(),epoch=2),a.output/'latest.pt')
    saved=torch.load(a.output/'latest.pt',weights_only=True)
    resumed=copy.deepcopy(template);resumed.load_state_dict(saved['model'],strict=True)
    opt3=build_optimizer(resumed);opt3.load_state_dict(saved['optimizer']);epoch(resumed,opt3,3)
    error=max(float((v-resumed.state_dict()[k]).abs().max()) for k,v in uninterrupted.state_dict().items())
    assert error==0
    report=dict(passed=True,protocol_version=VERSION,train_only_scaler=True,constant_and_nonfinite_handled=True,
        legacy_checkpoint_rejected=True,raw_descriptor_schema_verified=True,five_split_fit_positions_verified=True,
        optimizer='Adam',lr=.001,epoch_boundary_resume_parameter_max_error=error,formal_training_resumed=False)
    save_json(a.output/'acceptance.json',report);print(report)


if __name__=='__main__':main()
