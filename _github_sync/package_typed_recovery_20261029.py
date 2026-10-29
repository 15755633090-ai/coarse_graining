import argparse
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',action='store_true')
    p.add_argument('--start',type=int,default=0)
    p.add_argument('--limit',type=int,default=3)
    p.add_argument('--offset',type=int,default=0)
    p.add_argument('--chars',type=int)
    args=p.parse_args()
    module=ROOT/'motif/adjacency_diffusion'
    files=[module/name for name in ('typed_bonds.py','typed_train.py','typed_refined.py','test_typed_bonds.py','test_typed_refined.py',
                                    'TYPED_RECOVERY_REVISION_20261029.md','README.md')]
    files += [ROOT/'_github_sync'/name for name in ('check_typed_recovery_20261029.py',
              'summarize_typed_recovery_20261029.py','package_typed_recovery_20261029.py')]
    run=module/'runs/typed_recovery_comparison_20261029'
    files += [p for p in run.iterdir() if p.is_file() and (p.suffix=='.json' or p.name=='REPORT.md')]
    files += list(run.glob('*/history.json'))
    files += list((module/'runs/typed_refined_cli_smoke_20261029').rglob('*.json'))
    bounded=module/'runs/typed_bounded_noise_20261029'
    if bounded.exists():
        files += [p for p in bounded.iterdir() if p.is_file() and (p.suffix=='.json' or p.name=='REPORT.md')]
        files += list(bounded.glob('*/history.json'))
    files=sorted(files)
    if args.manifest:
        result={'files':[{'path':p.relative_to(ROOT).as_posix(),
                         'chars':len(p.read_text(encoding='utf-8-sig')),
                         'sha256':hashlib.sha256(p.read_text(encoding='utf-8-sig').encode()).hexdigest()} for p in files],
                'local_only_checkpoints':[{'path':p.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in run.glob('*/encoder.pt')]}
        print(json.dumps(result))
    else:
        print(json.dumps([{'path':p.relative_to(ROOT).as_posix(),'mode':'100644','type':'blob',
                          'content':p.read_text(encoding='utf-8-sig')[args.offset:args.offset+args.chars if args.chars is not None else None]} for p in files[args.start:args.start+args.limit]],ensure_ascii=True))


if __name__=='__main__':
    main()
