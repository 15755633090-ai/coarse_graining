"""Synthetic histories test gates only; no molecular test data are read."""
import copy
import hashlib
import json
from pathlib import Path
from .data import save_json
from .hmg_study_controls import convergence_record,budget_decision,seal_panel
from .evaluate_hmg_sealed import verify_seal

def main():
    checks=[]
    def reject(name,call):
        try:call()
        except (ValueError,KeyError):checks.append(name)
        else:raise AssertionError('Accepted '+name)
    history=[dict(epoch=i,valid_rmse=1. if i==1 else 2.) for i in range(1,52)]
    stopped=convergence_record(history,60)
    assert stopped['stop_reason']=='early_stop_criterion'
    capped=convergence_record(history[:2],2)
    reject('missing validation epochs',lambda:convergence_record([history[0],history[2]],60))
    reject('history beyond cap',lambda:convergence_record(history,2))
    assert budget_decision([capped],[2,60])['cap']==60
    reject('unequal caps',lambda:budget_decision([capped,stopped],[2,60]))
    choices=dict(target_scaling='raw',scheduler='constant',budget_ladder=[2,60])
    plan=dict(cases=[dict(seed=2020,case_id='native/A')],prepared_fingerprint='synthetic',**choices)
    artifact=dict(seed=2020,case_id='native/A',prepared_fingerprint='synthetic',test_evaluated=False,
                  shared_training_choices=choices,convergence=stopped,checkpoint_sha256='synthetic-only')
    seal=seal_panel(plan,[artifact]);verify_seal(seal);checks.append('valid panel seals')
    reject('missing panel',lambda:seal_panel(plan,[]))
    reject('duplicate panel',lambda:seal_panel(plan,[artifact,artifact]))
    bad=copy.deepcopy(artifact);bad['test_evaluated']=True
    reject('test already evaluated',lambda:seal_panel(plan,[bad]))
    bad=copy.deepcopy(artifact);bad['convergence']=capped
    reject('budget extension required',lambda:seal_panel(plan,[bad]))
    bad=copy.deepcopy(plan);bad['target_scaling']=None
    reject('unresolved plan',lambda:seal_panel(bad,[artifact]))
    bad=copy.deepcopy(artifact);bad['prepared_fingerprint']='other'
    reject('wrong inputs',lambda:seal_panel(plan,[bad]))
    bad=copy.deepcopy(seal);bad['cases'][0]['selected_epoch']=2
    reject('changed seal',lambda:verify_seal(bad))
    save_json(Path('motif/runs/hmg_design_v3_seal_checks_20261026.json'),
              dict(passed=True,checks=checks,synthetic_histories_only=True,test_evaluated=False,formal_training_resumed=False))
    print('Sealing gate checks passed; no test data read.')

if __name__=='__main__':main()
