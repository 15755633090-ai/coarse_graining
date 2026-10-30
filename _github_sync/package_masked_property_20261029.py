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
    files=[module/name for name in ('MASKED_RECOVERY_20261029.md','MASKED_PROPERTY_PROTOCOL_20261029.md','README.md')]
    files += [ROOT/'_github_sync/check_masked_property_20261029.py',Path(__file__).resolve()]
    files += [ROOT/'_github_sync/finalize_masked_property_20261029.py']
    weights=[]
    for name in ('masked_property_tool_smoke_20261029','masked_property_comparison_20261029'):
        folder=module/'runs'/name
        files += [path for path in folder.rglob('*.json') if 'frozen_source' not in path.parts]
        files += list(folder.glob('REPORT.md'))
        weights += [path for path in folder.rglob('*.pt') if 'frozen_source' not in path.parts]
    files=sorted(set(files))
    if args.manifest:
        print(json.dumps({'files':[{'path':path.relative_to(ROOT).as_posix(),
            'chars':len(path.read_text(encoding='utf-8-sig')),
            'sha256':hashlib.sha256(path.read_text(encoding='utf-8-sig').encode()).hexdigest()} for path in files],
            'local_only_checkpoints':[{'path':path.relative_to(ROOT).as_posix(),'bytes':path.stat().st_size,
                'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for path in weights]}))
    else:
        print(json.dumps([{'path':path.relative_to(ROOT).as_posix(),'mode':'100644','type':'blob',
            'content':path.read_text(encoding='utf-8-sig')[args.offset:args.offset+args.chars if args.chars is not None else None]}
            for path in files[args.start:args.start+args.limit]],ensure_ascii=True))


if __name__=='__main__':
    main()
