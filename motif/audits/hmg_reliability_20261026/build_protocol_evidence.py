"""Separate paper facts, classification-reference facts and unresolved regression choices."""
import ast
import hashlib
import json
from pathlib import Path

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]
repo=ROOT/'research_code_audit/HMG-MRL'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
config=ast.parse((repo/'config.py').read_text(encoding='utf-8-sig'))
cls=next(n for n in config.body if isinstance(n,ast.ClassDef) and n.name=='Config')
values={}
for node in cls.body:
    if isinstance(node,ast.Assign):
        try:value=ast.literal_eval(node.value)
        except (ValueError,TypeError):continue
        for target in node.targets:
            if isinstance(target,ast.Name):values[target.id]=value
entry=ast.parse((repo/'run_main.py').read_text(encoding='utf-8-sig'))
grid=next(ast.literal_eval(n.value) for n in entry.body if isinstance(n,ast.Assign)
          and any(isinstance(t,ast.Name) and t.id=='hyperparameter_setting' for t in n.targets))
search=json.loads((OUT/'public_materials_search.json').read_text(encoding='utf-8'))
scope=json.loads((OUT/'public_repository_scope.json').read_text(encoding='utf-8-sig'))
assert values['epochs']==values['max_epochs']==100
assert values['lr_scheduler_type']=='noam' and values['warmup_epochs']==2
assert grid['seed_number']==[2020,2021,2022,2023,2024]
code={key:values[key] for key in ('task_name','tasks','epochs','max_epochs','batch_size','warmup_epochs',
    'max_lr','final_lr','early_roc_epochs','early_loss_epochs','lr_scheduler_type','weight_decay','temperature','ct_loss_coff')}
assert code['temperature']==.15 and code['ct_loss_coff']==.15
code['grid_learning_rate']=grid['learning_rate'];code['grid_seeds']=grid['seed_number']
code.update(noam_init_lr=1e-4,noam_step_timing='after each training batch',
    stopping='both ROC stagnation >50 and loss stagnation >20',
    target_loss='weighted CrossEntropyLoss; classification reference, not regression')
matrix=[
 dict(setting='regression_alignment_hyperparameters',paper='supplement section 5: alpha=.15 and tau=.30 for all regression datasets',
      public_code='classification default alpha=.15, tau=.15',
      conclusion='v2 explicitly applies tau=.30 before model construction; runtime InfoNCE must be checked',status='paper_regression_confirmed'),
 dict(setting='regression_prediction_loss',paper='MSE, equation 12 explanation',public_code='classification CrossEntropyLoss',
      conclusion='MSE confirmed; raw vs train-standardized regression targets not specified',status='partial'),
 dict(setting='regression_target_scaling',paper='no explicit target/label scaling statement found in inspected full text and supplement',
      public_code='no regression training entry found',conclusion='must remain an explicit assumption',status='unresolved'),
 dict(setting='regression_scheduler',paper='initial learning rate 1e-3, scheduler not specified',
      public_code='classification Noam: init 1e-4, peak 1e-3, final 1e-4, warmup 2 epochs',
      conclusion='public Noam is known; not evidence of the regression schedule',status='classification_reference_only'),
 dict(setting='regression_max_epochs',paper='patience 50; no maximum epoch budget identified',public_code='classification epochs=100 and max_epochs=100',
      conclusion='100 is explicit in public classification config, not confirmed for regression',status='classification_reference_only'),
 dict(setting='paper_scaffold_rows',paper='8:1:1 scaffold, five runs',public_code='wrapper calls external scaffold_split; train/valid repool uses random_state=8',
      conclusion='exact implementation and five row lists not supplied; new strict splits are project splits',status='unresolved'),
 dict(setting='single_granularity_definition',paper='supplement section 12 gives single-granularity and capacity-matched variants',
      public_code='only full model and classification entry; variant scripts not found',
      conclusion='current fusion-subset variants are not verified author variants',status='unresolved'),
 dict(setting='fusion_formula_vs_code',paper='equation 11 displays weighted sum',
      public_code='weighted context plus unweighted view-sum residual, then activation',
      conclusion='preserve official forward; ask whether formula is abbreviated',status='clarification_needed'),
 dict(setting='CLS_construction',paper='describes a learnable CLS token',
      public_code='first token initialized by whole-molecule atom/bond statistics, then learned projections/attention',
      conclusion='do not invent a free nn.Parameter CLS; ask intended token construction',status='clarification_needed')]
record=dict(date='2026-10-26',official_commit='6ad69261eb1af36c8c0b96594d79a7639b969223',
    paper_url='https://academic.oup.com/bioinformatics/article/42/8/btag600/8758503',
    primary_text_characters=search['article_characters'],supplement_pages=search['supplement_pages'],
    target_scaling_keyword_hits=len(search['search']['target_scaling']),
    public_reference=code,official_history=scope['commits'],official_history_changed_files=scope['changed_files_since_initial'],
    branches_checked=[b['name'] for b in scope['branches']],releases_checked=len(scope['releases']),issues_checked=len(scope['issues']),
    source_sha256={str(p.relative_to(repo)):sha(p) for p in
        (repo/'config.py',repo/'run_main.py',repo/'other_utils.py',repo/'model/bipartite_transformer.py')},
    settings=matrix,exact_paper_regression_recipe_recovered=False,formal_training_remains_paused=True,
    limitation='Keyword absence and lack of public scripts do not establish what was used privately; no author contact.')
(OUT/'protocol_evidence_matrix.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(known_public_max_epochs=code['max_epochs'],known_public_scheduler=code['lr_scheduler_type'],
    exact_paper_regression_recipe_recovered=False,settings=len(matrix),formal_training_remains_paused=True)))
