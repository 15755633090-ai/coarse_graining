import argparse
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',action='store_true')
    p.add_argument('--start',type=int,default=0)
    p.add_argument('--limit',type=int,default=1)
    p.add_argument('--offset',type=int,default=0)
    p.add_argument('--chars',type=int)
    args=p.parse_args()
    module=ROOT/'motif/adjacency_diffusion'
    files=[module/name for name in ('typed_train.py','test_typed_refined.py','README.md','TRANSFER_DIAGNOSIS_20261029.md')]
    files += [ROOT/'_github_sync'/name for name in ('analyze_typed_transfer_20261029.py','check_frozen_typed_heads_20261029.py',
        'check_typed_full_budget_20261029.py','check_typed_property_20261029.py','summarize_typed_transfer_20261029.py',
        'finalize_typed_transfer_20261029.py','package_typed_diagnostics_20261029.py')]
    weights=[]
    for name in ('typed_transfer_diagnostics_20261029','typed_frozen_head_diagnostics_20261029',
                 'typed_full_budget_diagnostics_20261029','typed_low_encoder_lr_20261029','typed_low_lr_tool_smoke_20261029'):
        folder=module/'runs'/name
        files += [path for path in folder.rglob('*.json') if 'frozen_source' not in path.parts]
        files += list(folder.glob('REPORT.md'))
        weights += [path for path in folder.rglob('*.pt') if 'frozen_source' not in path.parts]
    files=sorted(set(files))
    if args.manifest:
        print(json.dumps({'files':[{'path':path.relative_to(ROOT).as_posix(),
            'chars':len(path.read_text(encoding='utf-8-sig')),
            'sha256':hashlib.sha256(path.read_text(encoding='utf-8-sig').encode()).hexdigest()} for path in files],
            'local_only_tensors':[{'path':path.relative_to(ROOT).as_posix(),'bytes':path.stat().st_size,
                'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for path in weights]}))
    else:
        print(json.dumps([{'path':path.relative_to(ROOT).as_posix(),'mode':'100644','type':'blob',
            'content':path.read_text(encoding='utf-8-sig')[args.offset:args.offset+args.chars if args.chars is not None else None]}
            for path in files[args.start:args.start+args.limit]],ensure_ascii=True))


if __name__=='__main__':
    main()
