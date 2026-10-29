import argparse
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',action='store_true')
    parser.add_argument('--start',type=int,default=0)
    parser.add_argument('--limit',type=int,default=3)
    args=parser.parse_args()
    module=ROOT/'motif/adjacency_diffusion'
    files=[module/name for name in ('typed_bonds.py','typed_train.py','test_typed_bonds.py','TYPED_BONDS_20261029.md','README.md')]
    files.append(Path(__file__).resolve())
    run=module/'runs/typed_bond_smoke_20261029'
    files+=list(run.rglob('*.json'))
    files=sorted(files)
    if args.manifest:
        rows=[{'path':p.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(p.read_text(encoding='utf-8-sig').encode()).hexdigest()} for p in files]
        weights=[{'path':p.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in run.rglob('*.pt')]
        print(json.dumps({'files':rows,'local_only_checkpoints':weights}))
    else:
        print(json.dumps([{'path':p.relative_to(ROOT).as_posix(),'mode':'100644','type':'blob','content':p.read_text(encoding='utf-8-sig')} for p in files[args.start:args.start+args.limit]],ensure_ascii=True))


if __name__=='__main__':
    main()
