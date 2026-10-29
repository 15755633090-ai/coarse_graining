"""Selected source and compact experiment records; never upload datasets/weights."""
import argparse
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
module=ROOT/'motif/adjacency_diffusion'


def paths():
    files=list(module.glob('*.py'))+list(module.glob('*.md'))
    files += [ROOT/'_github_sync'/name for name in (
        'search_adjacency_rescue_20261029.py','check_rescued_property_20261029.py',
        'check_posterior_20261029.py','check_likelihood_baseline_20261029.py',
        'summarize_posterior_20261029.py','package_rescue_revision_20261029.py')]
    for run in ('rescue_screen_20261029','rescue_property_20261029','posterior_comparison_20261029'):
        folder=module/'runs'/run
        files += [p for p in folder.iterdir() if p.is_file() and (p.suffix=='.json' or p.name=='REPORT.md')]
    return sorted(set(files))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--start',type=int,default=0)
    parser.add_argument('--limit',type=int,default=3)
    parser.add_argument('--manifest',action='store_true')
    args=parser.parse_args()
    files=paths()
    if args.manifest:
        rows=[]
        for path in files:
            data=path.read_text(encoding='utf-8-sig').encode()
            rows.append({'path':path.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)})
        checkpoints=[]
        for run in ('rescue_screen_20261029','rescue_property_20261029','posterior_comparison_20261029'):
            for path in (module/'runs'/run).rglob('*.pt'):
                if 'frozen_source' not in path.parts:
                    checkpoints.append({'path':path.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size})
        print(json.dumps({'files':rows,'local_only_checkpoints':checkpoints}))
    else:
        print(json.dumps([{'path':p.relative_to(ROOT).as_posix(),'mode':'100644','type':'blob',
                           'content':p.read_text(encoding='utf-8-sig')} for p in files[args.start:args.start+args.limit]],ensure_ascii=True))


if __name__=='__main__':
    main()
