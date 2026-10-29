"""Non-neural noise-evidence baselines, evaluated on a frozen experiment's folds."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from _github_sync import search_adjacency_rescue_20261029 as base
from motif.adjacency_diffusion.diffusion import Schedule
from motif.adjacency_diffusion.posterior import likelihood_log_odds
from motif.adjacency_diffusion.reparameterizations import degree_prior_baseline


class EvidenceOnly(torch.nn.Module):
    def __init__(self,density,degree=False):
        super().__init__()
        self.schedule=Schedule()
        self.density=density
        self.degree=degree

    def predict_noise(self,condition,noisy,t):
        alpha=self.schedule.alpha_bar.to(noisy.device)[t].to(noisy.dtype)[:,None,None]
        prior=degree_prior_baseline(condition,self.density) if self.degree else torch.full_like(noisy,self.density)
        # Degree probabilities are an approximate conditional prior, not an oracle.
        logits=torch.logit(prior.clamp(1e-6,1-1e-6))
        return (logits+likelihood_log_odds(noisy,alpha,condition.pairs))*condition.pairs


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    args=p.parse_args()
    plan=json.loads((args.run/'plan.json').read_text(encoding='utf-8'))
    plan['device']='cpu'
    torch.set_num_threads(2)
    density=base.positive_rate(base.GraphDataset(plan['train_smiles']))
    rows={}
    for name,degree in [('density_plus_likelihood',False),('degree_plus_likelihood',True)]:
        model=EvidenceOnly(density,degree)
        rows[name]={fold:base.evaluate(model,base.GraphDataset(plan[key]),{'prior':False,'mode':'bounded_A'},plan,density)
                    for fold,key in [('development','development_smiles'),('confirmation','confirmation_smiles')]}
    base.atomic_write(args.run/'nonneural_likelihood_baselines.json',{
        'baselines':rows,'learned_parameters':0,'training_positive_rate':density,
        'test_evaluated':False,'same_noise_seeds_and_folds_as_neural_comparison':True,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    for name,row in rows.items():
        print(name,'confirm_mid_ratio',base.mid_score(row['confirmation']),
              'AP500',row['confirmation']['500']['average_precision'])


if __name__=='__main__':
    main()
