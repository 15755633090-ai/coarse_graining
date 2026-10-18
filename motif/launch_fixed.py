"""Run the three matched arms for one paper, fail on initial-backbone mismatch."""
import argparse
from pathlib import Path
import subprocess
import sys
import torch
from .data import save_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--paper', choices=['HimNet','HMG-MRL'], required=True)
    for name in ('paper-root', 'data', 'checkpoint', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--hmg-feature-cache', type=Path)
    p.add_argument('--seeds', type=int, nargs='+', default=[0,1,2])
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--encoder-lr', type=float, default=1e-5)
    p.add_argument('--smoke-per-split', type=int, default=0)
    p.add_argument('--hidden', type=int)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    save_json(a.output/'protocol.json',{
        'comparison':'initialization under the same training protocol; not separately optimized random baseline',
        'lr':a.lr,'encoder_lr':a.encoder_lr,'auxiliary_weight':0,
        'baseline':'original architecture under unified protocol; not full paper reproduction',
        'seeds':a.seeds})
    outcomes = []
    for seed in a.seeds:
        reference = None
        for arm in ('original','random','pretrained'):
            out = a.output/f'seed_{seed}'/arm
            cmd = [sys.executable,'-m','motif.train_paper','--fixed-topology','--fusion','residual','--aux-weight','0']
            options = {'paper':a.paper, 'paper-root':a.paper_root, 'data':a.data,
                       'checkpoint':a.checkpoint, 'output':out, 'arm':arm,'seed':seed,
                       'epochs':a.epochs, 'patience':a.patience, 'batch-size':a.batch_size,
                       'lr':a.lr,'encoder-lr':a.encoder_lr,'smoke-per-split':a.smoke_per_split,
                       'hidden':a.hidden,'hmg-feature-cache':a.hmg_feature_cache}
            for key,value in options.items():
                if value is not None: cmd.extend(['--'+key,str(value)])
            subprocess.run(cmd, check=True)
            state = torch.load(out/'initial_backbone.pt', map_location='cpu', weights_only=True)
            if reference is None: reference = state
            elif state.keys()!=reference.keys() or any(not torch.equal(state[k],reference[k]) for k in state):
                raise RuntimeError(f'Initial backbone differs: seed={seed}, arm={arm}')
            outcomes.append({'seed':seed,'arm':arm,'output':str(out),'initial_backbone_matched':True})
            save_json(a.output/'completed.json',outcomes)


if __name__=='__main__':main()
