"""Package completed fullgraph experiment for the existing project repository."""
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT/'coarse_graining1'
RUN=ROOT/'motif/runs/fullgraph_formal_seed0_20261016'
DEST=REPO/'results/fullgraph_formal_seed0_20261016'

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def archive(path,entries):
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for source,name in entries:z.write(source,name)
    # GitHub rejects individual blobs over 100 MB. Keep all bytes in parts.
    if path.stat().st_size>90*1024**2:
        with path.open('rb') as f:
            i=1
            while True:
                b=f.read(64*1024**2)
                if not b:break
                path.with_name(path.name+f'.part{i:03d}').write_bytes(b);i+=1
        path.unlink()

def main():
    DEST.mkdir(parents=True,exist_ok=True)
    for f in (ROOT/'motif').iterdir():
        if f.is_file() and f.suffix in ('.py','.md','.json','.txt'):
            shutil.copy2(f,REPO/'motif'/f.name)
    legacy=REPO/'扩散GNN/01_扩散预训练/bond_diffusion'
    legacy.mkdir(parents=True,exist_ok=True)
    for f in (ROOT/'扩散GNN/01_扩散预训练/bond_diffusion').glob('*.py'):
        shutil.copy2(f,legacy/f.name)
    for f in RUN.rglob('*'):
        if f.is_file() and f.suffix not in ('.pt','.pyc') and '__pycache__' not in f.parts:
            target=DEST/f.relative_to(RUN);target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(f,target)
    artifacts=DEST/'artifacts';artifacts.mkdir(exist_ok=True)
    for stage in ('him_pretrain','him_original','him_pretrained','hmg_pretrain','hmg_original','hmg_pretrained'):
        archive(artifacts/(stage+'.zip'),[(p,p.relative_to(RUN).as_posix()) for p in (RUN/stage).glob('*.pt')])
    sources=[ROOT/'motif/data/lipo.pt',ROOT/'motif/data/lipo.json',
             ROOT/'motif/runs/hmg_features_finite_20261016/features.pickle',
             ROOT/'motif/runs/hmg_features_finite_20261016/manifest.json']
    archive(artifacts/'inputs.zip',[(p,p.relative_to(ROOT).as_posix()) for p in sources])
    entries=[]
    for base in (ROOT/'research_code_audit/HMG-MRL',REPO/'HimNet_reproduction_20261013/windows'):
        for f in base.rglob('*'):
            if f.is_file() and '.git' not in f.parts and '__pycache__' not in f.parts and (f.suffix in ('.py','.yaml','.yml','.json','.md','.txt') or f.name.startswith('LICENSE')):
                entries.append((f,f.relative_to(ROOT).as_posix()))
    archive(artifacts/'paper_sources.zip',entries)
    # Preserve the independent controlled experiments cited in the discussion.
    for name in ('gine_motif_readout_20261014','gine_motif_validation_20261014','motif_controls_20261015'):
        target=DEST/'related_controls'/name
        control_entries=[]
        for f in (REPO/name).rglob('*'):
            if f.is_file() and f.suffix in ('.py','.md','.json','.csv') and not any(p in ('__pycache__','smoke','smoke_context') for p in f.relative_to(REPO/name).parts):
                control_entries.append((f,f.relative_to(REPO).as_posix()))
        archive(artifacts/(name+'.zip'),control_entries)
        for f in (REPO/name).rglob('RESULTS.md'):
            if 'smoke' not in str(f):
                out=target/f.relative_to(REPO/name);out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,out)
    manifest={str(p.relative_to(DEST)).replace('\\','/'):{'bytes':p.stat().st_size,'sha256':digest(p)}
              for p in DEST.rglob('*') if p.is_file() and p.name!='FILES_SHA256.json'}
    (DEST/'FILES_SHA256.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    for p in artifacts.iterdir():print(p.name,round(p.stat().st_size/1024**2,2),'MiB',flush=True)

if __name__=='__main__':main()
