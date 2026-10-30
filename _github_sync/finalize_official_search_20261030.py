"""Independently recompute CSV metrics and retain every mature-method outcome."""
import csv,json,hashlib,math,statistics,tarfile,io
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];central=ROOT/'motif/official_method_search'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
def check_csv(folder,datafolder,name,row):
    for split in ['train','valid']:
        with (folder/f'{split}_predictions.csv').open(encoding='utf-8') as f:pred=list(csv.DictReader(f))
        with (datafolder/f'{name}_{split}.csv').open(encoding='utf-8') as f:truth=list(csv.DictReader(f))
        assert [p['sample_id'] for p in pred]==[p['sample_id'] for p in truth]
        assert all(abs(float(p['target'])-float(t['target']))<1e-8 for p,t in zip(pred,truth))
        error=[float(p['prediction'])-float(t['target']) for p,t in zip(pred,truth)]
        rmse=math.sqrt(statistics.mean(e*e for e in error));mae=statistics.mean(abs(e) for e in error)
        assert abs(rmse-row[split]['rmse'])<1e-8 and abs(mae-row[split]['mae'])<1e-8

runs={'chemeleon_smoke':ROOT/'motif/official_chemeleon/runs/smoke_20261030',
      'chemeleon_formal':ROOT/'motif/official_chemeleon/runs/comparison_20261030',
      'rich_formal':ROOT/'motif/official_chemprop_rich/runs/comparison_20261030'}
audits=[];summary={};archive=central/'frozen_sources_20261030.tar.xz'
with tarfile.open(archive,'w:xz',preset=6) as z:
    for tag,out in runs.items():
        plan=read(out/'plan.json');status=read(out/'status.json');result=read(out/'results.json');assert status['state']=='complete'
        assert len(result['runs'])==len(plan['datasets'])*len(plan['seeds'])
        for rel,h in plan['source_sha256'].items():
            path=out/'frozen_source'/rel;assert sha(path)==h
            raw=path.read_bytes();entry=tarfile.TarInfo(tag+'/frozen_source/'+rel);entry.size=len(raw);entry.mtime=0;entry.mode=0o644;z.addfile(entry,io.BytesIO(raw))
        for rel,h in plan['data_sha256'].items():
            assert sha(out/rel)==h
            raw=(out/rel).read_bytes();entry=tarfile.TarInfo(tag+'/'+rel);entry.size=len(raw);entry.mtime=0;entry.mode=0o644;z.addfile(entry,io.BytesIO(raw))
        raw=(out/'plan.json').read_bytes();entry=tarfile.TarInfo(tag+'/plan.json');entry.size=len(raw);entry.mtime=0;entry.mode=0o644;z.addfile(entry,io.BytesIO(raw))
        for r in result['runs']:
            folder=out/r['dataset']/f"seed_{r['seed']}";hist=read(folder/'history.json');assert len(hist)==plan['epochs']
            config=read(folder/'config.json');check_csv(folder,out/'data',r['dataset'],r)
            assert abs(min(h['val_loss'] for h in hist)-r['valid']['rmse']**2/config['target_scale'][0]**2)<1e-4
            assert sha(folder/'checkpoints/best.ckpt')==r['best_checkpoint_sha256']
        audits.append({'run':tag,'source_data_histories_checkpoints_and_prediction_csvs_passed':True,'models':len(result['runs'])})
        if tag!='chemeleon_smoke':summary[tag]=result['aggregate']
    adaptive=['run_frozen_chemeleon_ridge_20261030.py','recheck_chemeleon_ridge_float64_20261030.py','export_normalized_descriptors_20261030.py',
              'run_descriptor_xgboost_20261030.py','verify_descriptor_svr_20261030.py','check_official_search_data_20261030.py',
              'finalize_official_search_20261030.py','package_official_search_20261030.py']
    for name in adaptive:
        p=ROOT/'_github_sync'/name
        if not p.exists():continue
        raw=p.read_bytes();entry=tarfile.TarInfo('adaptive_sources/'+name);entry.size=len(raw);entry.mtime=0;entry.mode=0o644;z.addfile(entry,io.BytesIO(raw))
    for p in sorted(central.glob('*.py')):
        raw=p.read_bytes();entry=tarfile.TarInfo('inference_sources/'+p.name);entry.size=len(raw);entry.mtime=0;entry.mode=0o644;z.addfile(entry,io.BytesIO(raw))
    old=runs['rich_formal']/'descriptor_xgboost/source_before_io_fix.py';raw=old.read_bytes();entry=tarfile.TarInfo('adaptive_sources/xgboost_before_io_fix.py');entry.size=len(raw);entry.mtime=0;entry.mode=0o644;z.addfile(entry,io.BytesIO(raw))
chemel=runs['chemeleon_formal'];rich=runs['rich_formal']
ridge=read(chemel/'frozen_ridge/results.json');double=read(chemel/'frozen_ridge_float64/results.json')
assert double['source_script_sha256']==sha(ROOT/'_github_sync/recheck_chemeleon_ridge_float64_20261030.py')
for r in ridge['runs']:check_csv(chemel/'frozen_ridge'/r['dataset'],chemel/'data',r['dataset'],r)
for r in double['runs']:check_csv(chemel/'frozen_ridge_float64'/r['dataset'],chemel/'data',r['dataset'],r)
svr=read(rich/'svr_results.json')
for r in svr['runs']:
    check_csv(rich/r['dataset']/'svr',rich/'data',r['dataset'],r)
    assert read(rich/r['dataset']/'svr/verification.json')['replay_passed']
xgb=read(rich/'descriptor_xgboost/results.json');xplan=read(rich/'descriptor_xgboost/plan.json')
assert xplan['source_script_sha256']==sha(ROOT/'_github_sync/run_descriptor_xgboost_20261030.py')
assert xplan['io_revision']['original_source_sha256']==sha(rich/'descriptor_xgboost/source_before_io_fix.py')
for r in xgb['runs']:
    folder=rich/'descriptor_xgboost'/r['dataset']
    # Reuse the same checker through temporary in-memory filename-independent rows.
    for split in ['train','valid']:
        with (folder/f"{split}_seed_{r['seed']}_predictions.csv").open(encoding='utf-8') as f:pred=list(csv.DictReader(f))
        with (rich/'data'/f"{r['dataset']}_{split}.csv").open(encoding='utf-8') as f:truth=list(csv.DictReader(f))
        assert [p['sample_id'] for p in pred]==[p['sample_id'] for p in truth]
        err=[float(p['prediction'])-float(t['target']) for p,t in zip(pred,truth)]
        assert abs(math.sqrt(statistics.mean(e*e for e in err))-r[split]['rmse'])<1e-8
    assert sha(folder/f"model_seed_{r['seed']}.json")==r['model_sha256']
old=read(ROOT/'motif/adjacency_diffusion/runs/masked_property_comparison_20261029/results.json')['aggregate']['scratch']
references={}
for name in ['esol','freesolv']:
    values=[read(p)['validation_metrics']['rmse'] for p in (ROOT/'model/results_formal/02_dmpnn'/name/'dmpnn').glob('seed_*/result.json')]
    references[name]={'n':len(values),'mean_rmse':statistics.mean(values),'sample_sd':statistics.stdev(values),'values':values}
summary.update(svr={r['dataset']:r['valid'] for r in svr['runs']},xgboost=xgb['aggregate'],
    frozen_ridge={r['dataset']:r['valid'] for r in ridge['runs']},frozen_ridge_float64={r['dataset']:r['valid'] for r in double['runs']},
    original_custom_esol=old,official_dmpnn_references=references)
write(central/'results_20261030.json',summary)
write(central/'acceptance_20261030.json',{'passed':True,'auditor_sha256':sha(Path(__file__)),'neural_runs':audits,
    'all_prediction_csv_metrics_recomputed':True,'source_archive_sha256':sha(archive),'svr_standalone_inference_replayed_rows':113,
    'test_evaluated':False,'data_verification':read(central/'data_verification.json')})
# Publish only the compact ESOL SVR, not the external foundation or all checkpoints.
model_archive=central/'esol_svr_20261030.tar.xz'
with tarfile.open(model_archive,'w:xz',preset=6) as z:
    for p in [rich/'esol/svr/model.joblib',rich/'esol/svr/verification.json']:
        raw=p.read_bytes();entry=tarfile.TarInfo(p.relative_to(ROOT).as_posix());entry.size=len(raw);entry.mtime=0;entry.mode=0o644;z.addfile(entry,io.BytesIO(raw))
write(central/'esol_svr_archive.json',{'path':model_archive.name,'sha256':sha(model_archive),'bytes':model_archive.stat().st_size,
    'raw_model_sha256':sha(rich/'esol/svr/model.joblib'),'extract_at':'project root','scope':'candidate ESOL predictor; repeated validation search, no test claim'})
print(json.dumps({'passed':True,'summary':summary,'published_model_bytes':model_archive.stat().st_size}))
