# CLEAN FORMAL CHECKPOINT

This directory contains the leakage-clean diffusion pretraining run.

- Status: `clean/formal`
- Completed epochs: 30
- Pretraining molecules: 423,114
- Downstream overlap audit: zero for train/validation/test across ESOL,
  FreeSolv, Lipo, BBBP, BACE, HIV, and Tox21
- Canonical checkpoint for every formal recovery, probe, and downstream run:

`outputs/ogb_clean/best.pt`

Do not substitute a checkpoint from `outputs/ogb_large`.
