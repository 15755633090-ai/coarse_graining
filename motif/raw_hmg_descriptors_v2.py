"""Standalone Torch-free raw descriptor worker; launch as a file, not -m motif."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--requests',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    spec=importlib.util.spec_from_file_location('hmg_protocol_v2_cpu',Path(__file__).with_name('hmg_protocol_v2.py'))
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    request=json.loads(a.requests.read_text(encoding='utf-8-sig'))
    values=[]
    for i,row in enumerate(request['molecules']):
        values.append(helper.raw_descriptors(row['smiles'],request['names']))
        if (i+1)%100==0:
            state=dict(phase='raw_descriptors',completed=i+1,total=len(request['molecules']))
            (a.output.parent/'progress.json').write_text(json.dumps(state),encoding='utf-8')
            print(state,flush=True)
    np.savez_compressed(a.output,raw=np.stack(values))


if __name__=='__main__':main()
