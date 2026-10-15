import argparse
from pathlib import Path
import torch
from .data import load_dataset,collate_graphs,to_device,save_json,sha
from .pretrain import load_encoder

def main():
    p=argparse.ArgumentParser(description='Export deterministic clean motif features; no generation')
    p.add_argument('--data',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    model,ck=load_encoder(args.checkpoint,args.device);model.eval();data=load_dataset(args.data)
    if ck['data_sha256']!=sha(args.data): raise ValueError('Cache provenance mismatch')
    features=[]
    with torch.no_grad():
        for start in range(0,len(data['motifs']),args.batch_size):
            b=to_device(collate_graphs(data['motifs'][start:start+args.batch_size]),args.device)
            features.append(model.encode(b)['motif'].cpu())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'vectors':torch.cat(features) if features else torch.empty(0,model.config['hidden']),
                'index':[{'row':g['row'],'method':g['method'],'index':g['index'],'members':g['members'],'split':g['split']}
                         for g in data['motifs']],
                'checkpoint_sha256':sha(args.checkpoint),'data_sha256':sha(args.data),
                'smoke':ck['smoke'],'t':0},args.output)
    save_json(args.output.with_suffix('.json'),{'motifs':len(data['motifs']),'features':model.config['hidden'],
                    'sha256':sha(args.output),'clean_encoding':True,'smoke':ck['smoke']})

if __name__=='__main__':main()
