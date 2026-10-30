"""Predict with the ESOL standard descriptor-SVR candidate trained on train only."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np,joblib
from rdkit import Chem
from chemprop.featurizers import V1RDKit2DNormalizedFeaturizer
ROOT=Path(__file__).resolve().parents[2]
MODEL_DIR=ROOT/'motif/official_chemprop_rich/runs/comparison_20261030/esol/svr'

def predict(smiles):
    receipt=json.loads((MODEL_DIR/'verification.json').read_text(encoding='utf-8'))
    path=MODEL_DIR/'model.joblib'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=receipt['model_sha256']:raise ValueError('SVR checkpoint hash mismatch')
    generator=V1RDKit2DNormalizedFeaturizer();features=[]
    for s in smiles:
        mol=Chem.MolFromSmiles(s)
        if mol is None:raise ValueError(f'Invalid SMILES: {s}')
        row=np.array(generator(mol),dtype=np.float64);row[np.isnan(row)]=0
        if row.shape!=(200,) or not np.isfinite(row).all():raise ValueError('Invalid descriptor vector')
        features.append(row)
    if not features:return np.empty(0)
    return joblib.load(path).predict(np.stack(features))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input-csv',type=Path,required=True);p.add_argument('--output-csv',type=Path,required=True);p.add_argument('--smiles-column',default='smiles');a=p.parse_args()
    if a.output_csv.exists():raise FileExistsError(a.output_csv)
    with a.input_csv.open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    values=predict([r[a.smiles_column] for r in rows])
    with a.output_csv.open('w',encoding='utf-8',newline='') as f:
        writer=csv.writer(f);writer.writerow([a.smiles_column,'predicted_log_solubility']);writer.writerows((r[a.smiles_column],float(v)) for r,v in zip(rows,values))
