"""One domain correction justified by measured descriptor transfer failure."""
import argparse,sys,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from _github_sync.run_chemical_anchor_20261030 import prepare,execute,read,write,sha,runner

def prepare_domain(out):
    prepare(out)
    plan=read(out/'plan.json')
    source,splits,_=runner.io.build_property_data(Path(plan['data_root']),'esol')
    plan['corpus']=[source.smiles[i] for i in splits['train']]
    assert len(plan['corpus'])==902
    plan['domain_correction']='training-domain self-pretraining with calculated targets; no measured property labels upstream'
    plan['heldout_policy']='only ESOL training structures902; no validation/test molecules or measured y used in upstream targets/scaling'
    plan['decision']='same objective, backbone and update budget; test whether training-domain knowledge transfers better than source-domain knowledge; no hyperparameter search'
    plan['original_chemical_run']=str(ROOT/'motif/adjacency_diffusion/runs/chemical_anchor_20261030')
    plan['related_data_evidence']='source descriptor MSE .054-.072 vs ESOL training structures1.918-2.277 under the same source normalization'
    # Freeze this preparation logic as well, before any training.
    rel=Path(__file__).relative_to(ROOT).as_posix();dest=out/'frozen_source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(Path(__file__),dest)
    plan['source_sha256'][rel]=sha(Path(__file__))
    write(out/'plan.json',plan)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True,type=Path);p.add_argument('--execute',action='store_true');a=p.parse_args()
    (execute if a.execute else prepare_domain)(a.output_dir.resolve())
