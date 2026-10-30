"""Serialize two local GPU experiments; no overlapping training jobs."""
import json,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parents[1]
status=root/'motif/adjacency_diffusion/runs/transfer_revision_20261030/status.json'
deadline=time.monotonic()+7200
while True:
    current=json.loads(status.read_text(encoding='utf-8'))
    if current['state']=='complete':break
    if current['state']=='failed':raise RuntimeError(current)
    if time.monotonic()>deadline:raise TimeoutError('First experiment did not finish in two hours')
    time.sleep(5)
out=root/'motif/adjacency_diffusion/runs/posterior_weighting_20261030'
subprocess.run([sys.executable,str(out/'frozen_source/_github_sync/check_posterior_weighting_20261030.py'),
                '--output-dir',str(out),'--execute'],cwd=root,check=True)
