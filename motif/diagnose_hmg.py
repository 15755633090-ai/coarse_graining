"""Read-only finite-input and original-model evaluation diagnostic."""
from pathlib import Path
import argparse
import numpy as np
import torch
from .data import load_dataset,save_json
from .paper_data import load_hmg_features,hmg_inputs,load_paper

def main():
    torch.set_num_threads(4)
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feature-cache',type=Path,default=root/'motif/runs/hmg_features_repaired_20261016/features.pickle')
    parser.add_argument('--output',type=Path,default=root/'motif/runs/hmg_nan_diagnostic_20261016')
    args=parser.parse_args();output=args.output;output.mkdir(exist_ok=True)
    data=load_dataset(root/'motif/data/lipo.pt')
    features,assemble=load_hmg_features(args.feature_cache,root/'research_code_audit/HMG-MRL')
    bad=[]
    for name,records in features.items():
        for key,value in records.items():
            if isinstance(value,np.ndarray) and not np.isfinite(value).all():
                bad.append({'field':name,'smiles':key,'nonfinite':int((~np.isfinite(value)).sum())})
    save_json(output/'input_scan.json',{'nonfinite_fields':bad,'labels_finite':all(np.isfinite(r['label']) for r in data['molecules'])})
    print({'nonfinite_inputs':len(bad)},flush=True)
    if bad:return
    cls,_,cfg=load_paper(root/'research_code_audit/HMG-MRL','HMG-MRL','cuda:0')
    valid=[r for r in data['molecules'] if r['split']=='valid']
    sample,_,_=hmg_inputs(valid[:2],features,assemble,'cuda:0')
    cfg.atom_dim_in=sample['x_atoms'].shape[-1];cfg.bond_dim_in=sample['x_bonds'].shape[-1]
    model=cls(cfg.atom_dim_in,cfg.bond_dim_in,1).cuda()
    initial=torch.load(root/'motif/runs/paper_structure_hmg_seed0_20261016/hmg_original/initial_backbone.pt',weights_only=True)
    model.load_state_dict(initial);model.eval()
    current=[];events=[]
    def hook(name):
        def check(module,inputs,output):
            def tensors(value):
                if isinstance(value,torch.Tensor):yield value
                elif isinstance(value,(list,tuple)):
                    for item in value:yield from tensors(item)
                elif isinstance(value,dict):
                    for item in value.values():yield from tensors(item)
            if any(t.is_floating_point() and not bool(torch.isfinite(t).all()) for t in tensors(output)):
                event={'module':name,'type':type(module).__name__,'rows':list(current)}
                events.append(event);save_json(output/'first_nonfinite.json',event)
                raise FloatingPointError(str(event))
        return check
    handles=[module.register_forward_hook(hook(name)) for name,module in model.named_modules() if name]
    try:
        with torch.no_grad():
            for start in range(0,len(valid),16):
                rows=valid[start:start+16];current[:]=[r['row'] for r in rows]
                inputs,_,_=hmg_inputs(rows,features,assemble,'cuda:0')
                out=model(**inputs)
                if not torch.isfinite(out[1]).all():raise FloatingPointError('Nonfinite final predictions')
        print('Initial model validation forward finite',flush=True)
    finally:
        for handle in handles:handle.remove()
        save_json(output/'forward_scan.json',{'events':events})

if __name__=='__main__':main()
