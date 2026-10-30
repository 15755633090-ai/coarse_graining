"""Audit and package the fixed direct-supervision architecture comparison."""
import json,hashlib,tarfile,io,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];module=ROOT/'motif/boundary_supervised'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
for name in ['smoke_20261030','comparison_20261030']:
    out=module/'runs'/name;plan=read(out/'plan.json');status=read(out/'status.json');result=read(out/'results.json')
    assert status['state']=='complete' and len(result['runs'])==len(plan['arms'])*len(plan['seeds'])
    for rel,h in plan['source_sha256'].items():assert sha(out/'frozen_source'/rel)==h,rel
    audited=[]
    for seed in plan['seeds']:
        initial=[]
        for arm in plan['arms']:
            folder=out/f'{arm}_seed_{seed}';config=read(folder/'config.json');history=read(folder/'history.json')
            completion=read(folder/'completion.json');assert len(history)==plan['epochs']
            assert abs(min(r['validation']['rmse'] for r in history)-completion['validation']['rmse'])<1e-6
            assert completion['checkpoint_replay_passed'];initial.append(config['shared_initial_hash'])
            audited.append({'arm':arm,'seed':seed,'best_sha256':sha(folder/'best.pt'),'latest_sha256':sha(folder/'latest.pt'),
                'history_sha256':sha(folder/'history.json'),'active_parameters':config['active_parameters'],'epochs':len(history)})
        assert initial[0]==initial[1]
    archive=out/'frozen_source.tar.xz'
    with tarfile.open(archive,'w:xz',preset=6) as z:
        for rel,h in sorted(plan['source_sha256'].items()):
            raw=(out/'frozen_source'/rel).read_bytes();entry=tarfile.TarInfo('frozen_source/'+rel);entry.size=len(raw);entry.mtime=0;entry.mode=0o644;z.addfile(entry,io.BytesIO(raw))
    write(out/'frozen_source_archive.json',{'path':'frozen_source.tar.xz','sha256':sha(archive),'source_sha256':plan['source_sha256']})
    write(out/'acceptance.json',{'passed':True,'scope':'frozen-source hashes, histories, runtime checkpoint replay assertions and paired initialization',
        'auditor_sha256':sha(Path(__file__)),'runs':audited,'test_evaluated':False})
    lines=['# Direct supervised architecture result','', '| seed | arm | validation RMSE | training RMSE at best | active parameters | best epoch | seconds |',
           '|---:|---|---:|---:|---:|---:|---:|']
    for r in result['runs']:lines.append(f"| {r['seed']} | {r['arm']} | {r['validation']['rmse']:.6f} | {r['train']['rmse']:.6f} | {r['active_parameters']} | {r['best_epoch']} | {r['elapsed_seconds']:.2f} |")
    lines+=['',str(result['aggregate']),'', 'Original prior supervised reference: .787109 +/- .028044, n3.',
        'New directed attention baseline and BRICS boundary combination differ in capacity. No independent component attribution.',
        'No pretraining, computed descriptors, external teacher, or fingerprint features. Reused validation, no test evaluation.',
        'Smoke runs are implementation checks, not performance evidence.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'acceptance':'passed','aggregate':read(module/'runs/comparison_20261030/results.json')['aggregate']}))
