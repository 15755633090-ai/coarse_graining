# MotiL research audit

Start with [第一阶段结果](研究第一阶段结果.md) and [论文与研究方向核对](MotiL核对与研究方向评估.md).

This snapshot contains structural diagnostics, an instance-preserving prototype,
and inference verification. No property-performance improvement is claimed.

## Source and runtime

- Original implementation: <https://github.com/Young0222/MotiL>, MIT license in
  `../LICENSE.txt`.
- `upstream/` contains the official raw main files downloaded for this audit.
  Exact file hashes are recorded in `upstream_sha256.json` and the diagnostic
  report. These files are a snapshot, not a commit-pinned repository.
- `../MotiL_micromolecule/chemprop/` contains the existing Windows-compatible
  local implementation used for inference; it differs from `upstream/`.
- The original paper is linked in the reports. Paper full text/PDFs, virtual
  environments, datasets, weights, and training output directories are omitted
  from this GitHub synchronization.

The existing tested runtime uses Python 3.9, PyTorch 2.8, RDKit, PyG, NumPy,
scikit-learn, pandas, scipy, tensorboardX and Unidecode. See
`../MotiL_micromolecule/environment_windows_micro.yml` and
`../requirements-local.txt`; the latter supplements an existing environment and
is not a complete clean-install specification.

## Running diagnostics

Download dataset CSVs from the official repository's
`MotiL_micromolecule/data/` directory into that same relative local location.
The structural audit reads SMILES only and never uses property labels.

```powershell
& '.\MotiL_local\.venv\Scripts\python.exe' '.\MotiL_local\research_audit\diagnose_motif_selection.py'
```

For the inference check, obtain the official checkpoint from
<https://github.com/Young0222/MotiL/blob/main/MotiL_micromolecule/dumped/pre-train/1-model/original_CMPN_0707_0800_12000th_epoch.pkl>
and place it at the same relative path under `MotiL_local/`.
Expected SHA256:
`38082cdccb14a7d1b2749181548a81253e70196940bf979ef0f983b3a47e0f78`.

```powershell
& '.\MotiL_local\.venv\Scripts\python.exe' '.\MotiL_local\research_audit\verify_instance_encoder.py'
```

Both commands execute diagnostics/inference only. The prototype intentionally
retains legacy padding pooling, resets message-passing depth for each instance,
and records overlap separately from true chemical bridge bonds. Its connection
metadata has not yet been used to train a region message-passing model.
