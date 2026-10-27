"""Validation-only stopping/budget decisions and sealed-test evaluation contract."""
import copy
import hashlib
import json
import math

import torch
from torch import nn

DESIGN_VERSION='hmg-design-v3'


def standardized_output_initialization(model,mean,std):
    """Preserve initial raw predictions under yhat_raw=std*yhat_scaled+mean."""
    if not math.isfinite(std) or std<=0:raise ValueError('Positive finite training std required')
    final=[m for m in model.postGNN_classifier.modules() if isinstance(m,nn.Linear)][-1]
    if final.out_features!=1 or final.bias is None:raise ValueError('Scalar affine output required')
    with torch.no_grad():
        final.weight.div_(std);final.bias.sub_(mean).div_(std)


def convergence_record(history,budget,patience=50,near_optimal_tolerance=.005):
    if not history:raise ValueError('No completed validation epochs')
    if not isinstance(budget,int) or budget<1 or patience<1:raise ValueError('Positive integer budget/patience required')
    if [r['epoch'] for r in history]!=list(range(1,len(history)+1)):
        raise ValueError('Validation history must contain consecutive epochs starting at 1')
    if len(history)>budget:raise ValueError('Validation history exceeds declared cap')
    if any(not math.isfinite(float(r['valid_rmse'])) for r in history):raise ValueError('Nonfinite validation history')
    best=min(history,key=lambda r:r['valid_rmse'])
    last=history[-1]['epoch'];criterion=last-best['epoch']>=patience
    capped=last>=budget
    reason='early_stop_criterion' if criterion else 'budget_capped' if capped else 'incomplete'
    near=[r['epoch'] for r in history if r['valid_rmse']<=best['valid_rmse']+near_optimal_tolerance]
    ordered=sorted(r['valid_rmse'] for r in history)
    return dict(stop_reason=reason,selected_epoch=best['epoch'],best_valid_rmse=best['valid_rmse'],
        last_epoch=last,budget=budget,patience=patience,hit_budget=capped,
        operational_stopping_criterion_satisfied=criterion,
        needs_budget_review=reason in ('budget_capped','incomplete'),
        mathematical_convergence_claim=False,near_optimal_tolerance=near_optimal_tolerance,
        near_optimal_epochs=near,selection_margin=(ordered[1]-ordered[0]) if len(ordered)>1 else None)


def budget_decision(records,ladder):
    """Same predeclared cap for every member; never takes test metrics."""
    if not ladder or ladder!=sorted(set(ladder)) or ladder[0]<1:raise ValueError('Strictly increasing budget ladder required')
    if not records:raise ValueError('Empty comparison panel')
    caps={r['budget'] for r in records}
    if len(caps)!=1:raise ValueError('Paired groups must have identical caps before comparison')
    cap=caps.pop()
    if cap not in ladder:raise ValueError('Unregistered budget cap')
    if any(r['stop_reason']=='incomplete' for r in records):
        return dict(action='finish_current_cap',cap=cap,seal_test=False)
    if any(r['needs_budget_review'] for r in records):
        index=ladder.index(cap)
        if index+1<len(ladder):
            return dict(action='extend_panel_cap',cap=ladder[index+1],apply_cap_to_all=True,
                        preserve_identical_early_stopping_rule=True,seal_test=False)
        return dict(action='budget_limited',cap=cap,seal_test=False)
    return dict(action='validation_complete',cap=cap,seal_test=True)


def seal_panel(plan,artifacts):
    """Reject missing/unbalanced panels; test selection is frozen before scoring."""
    expected={(c['seed'],c['case_id']) for c in plan['cases']}
    actual={(c['seed'],c['case_id']) for c in artifacts}
    if expected!=actual or len(actual)!=len(artifacts):raise ValueError('Missing/duplicate validation cases')
    if any(a.get('test_evaluated',False) for a in artifacts):raise ValueError('Test was read before panel sealing')
    if any(a['prepared_fingerprint']!=plan['prepared_fingerprint'] for a in artifacts):raise ValueError('Input provenance mismatch')
    choices={json.dumps(a['shared_training_choices'],sort_keys=True) for a in artifacts}
    if len(choices)!=1:raise ValueError('Unpaired target/scheduler/selection choices')
    selected_choices=artifacts[0]['shared_training_choices']
    for key in ('target_scaling','scheduler','budget_ladder'):
        if plan.get(key) is None or selected_choices.get(key)!=plan[key]:
            raise ValueError('Plan choices unresolved or changed: '+key)
    # Each seed is a paired panel, not an independent sample from another split.
    decisions={}
    for seed in sorted({c['seed'] for c in plan['cases']}):
        records=[a['convergence'] for a in artifacts if a['seed']==seed]
        decision=budget_decision(records,plan['budget_ladder']);decisions[str(seed)]=decision
        if not decision['seal_test']:
            permitted=decision['action']=='budget_limited' and plan.get('accept_budget_limited',False)
            if not permitted:raise ValueError('Validation budget panel not resolved: '+str(decision))
    result=dict(design_version=DESIGN_VERSION,sealed=True,prepared_fingerprint=plan['prepared_fingerprint'],
        decisions=decisions,cases=[dict(seed=a['seed'],case_id=a['case_id'],
        selected_epoch=a['convergence']['selected_epoch'],checkpoint_sha256=a['checkpoint_sha256']) for a in artifacts],
        result_scope='budget-limited' if any(v['action']=='budget_limited' for v in decisions.values()) else 'operational_early_stop',
        allows_information_redundancy_claim=False)
    result['seal_sha256']=hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest()
    return result


def complete_design_plan(prepared_fingerprint,seeds,budget_ladder=None):
    """A complete, explicit matrix. It schedules no process and scores no test."""
    from .hmg_fusion_ablation import ARMS
    cases=[]
    for seed in seeds:
        for study in ('native','noaux','common-head','common-head-noaux','isolated'):
            for arm in ARMS:
                cases.append(dict(seed=seed,case_id=f'{study}/{arm}',study=study,arm=arm,capacity_target=None))
        for study in ('single-direct','single-direct-noaux'):
            for arm in ('A','S','D'):
                cases.append(dict(seed=seed,case_id=f'{study}/{arm}',study=study,arm=arm,capacity_target=None))
        for study in ('common-head','common-head-noaux','isolated'):
            for combination in ('AS','AD','SD','ASD'):
                for arm in combination:
                    cases.append(dict(seed=seed,case_id=f'{study}/{arm}_matched_to_{combination}',
                        study=study,arm=arm,capacity_target=combination))
    return dict(design_version=DESIGN_VERSION,prepared_fingerprint=prepared_fingerprint,
        cases=cases,budget_ladder=budget_ladder,accept_budget_limited=False,
        target_scaling=None,scheduler=None,loss_units='raw target units; alpha=.15 independent of output scaling',
        regression_temperature=.30,formal_training_paused=True,
        exact_author_independent_variants=False,
        isolation_scope='encoder-path isolation, not disjoint chemical information',
        metadata_only=True,test_evaluated=False)
