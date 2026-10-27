"""Compare label/structure identities to the verified official dataset."""
import collections
import csv
import json
from pathlib import Path

import torch
from rdkit import Chem

ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
torch.set_num_threads(2)
data=torch.load(ROOT/'motif/data/lipo.pt',map_location='cpu',weights_only=False)
with (ROOT/'research_code_audit/HMG-MRL/data/Lipophilicity.csv').open(encoding='utf-8-sig',newline='') as stream:
    official=list(csv.DictReader(stream))
def canonical(smiles):return Chem.MolToSmiles(Chem.MolFromSmiles(smiles),isomericSmiles=True)
first=collections.Counter((canonical(r['smiles']),round(float(r['label']),8)) for r in data['molecules'])
second=collections.Counter((canonical(r['smiles']),round(float(r['exp']),8)) for r in official)
report=dict(local_rows=len(data['molecules']),official_rows=len(official),
    canonical_isomeric_smiles_and_labels_multiset_equal=first==second,
    unmatched_local_rows=sum((first-second).values()),unmatched_official_rows=sum((second-first).values()),
    comparison='structure-label multiset, labels rounded to 8 decimal places; does not validate the split')
(OUT/'dataset_identity.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False))
