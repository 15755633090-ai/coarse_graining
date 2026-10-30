"""Export the read-only diagnosis and grounded method selection for GitHub."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
paths=['_github_sync/diagnose_task_latent_20261030.py','_github_sync/package_task_diagnosis_20261030.py',
       'motif/adjacency_diffusion/TASK_LATENT_DIAGNOSTICS_20261030.json',
       'motif/adjacency_diffusion/FAILURE_DIAGNOSIS_AND_TRANSFER_20261030.md']
entries=[{'path':p,'mode':'100644','type':'blob','content':(ROOT/p).read_text(encoding='utf-8-sig')} for p in paths]
print(json.dumps(entries,ensure_ascii=True))
