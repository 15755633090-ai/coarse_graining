# Transfer the propagation backbone and reset the output projection

This follow-up changes parameter transfer, while retaining the original mean-readout architecture and parameter count. It is another adaptive experiment on the reused ESOL validation split, not independent confirmation.

## Hypothesis

The encoder's final `fuse` MLP maps internal/context states into the node vectors used by the edge decoder. It was optimized for reconstruction. It may carry recovery-specific scale and coordinates that are less useful to a fresh property head. The preceding scale diagnostic and unsuccessful graph-normalization/readout bundle motivate testing this hypothesis; they do not establish that `fuse` is defective.

Load the original masked pretrained encoder except keys beginning `fuse.`. Keep that MLP at the exact task-seed initialization used by the original scratch baseline. All internal/context propagation, atom/edge input layers, time embeddings and boundary gates are transferred. The property head is fresh and matches the original scratch head initialization. Do not add parameters, atom-count features, posterior weighting, a new loss or an additional pretraining stage.

## Frozen comparison

Runner `_github_sync/check_partial_transfer_20261030.py`, output `runs/partial_transfer_20261030`. Same original upstream checkpoints (pretraining seeds7/1/2), task seeds0/1/2, scaffold train/validation, target scaler, batch16, dropout0.1, Adam0.001, no weight decay, clip5 and60 full epochs. All three seeds run without selection.

Original scratch and full-transfer references are reused: architecture, initial encoder/head hashes, task RNG, data order and optimizer settings are unchanged. Runtime assertions verify every transferred tensor against its source and every reset tensor against the fresh initialization. Best task checkpoints must replay predictions. The default full transfer remains available. The public typed CLI exposes the alternative as `--reset-output-projection` only with `--stage finetune --pretrained ...`; its metadata records `pretrained_except_fuse`.

Two unit tests verify selective reset with unchanged capacity and unchanged default full loading. This is transfer behavior, not evidence of prediction gain. Final results must be compared with both original scratch0.78711 and original full masked transfer0.81446; only improvement over a weaker modified baseline would be insufficient.

## Completed result

Reset-fuse validation RMSE is0.84708 ±0.06304 (three seeds), versus full transfer0.81446 ±0.05114 and scratch0.78711 ±0.02804. All three reset-fuse seeds are worse than both respective full-transfer and scratch references. The hypothesis was not supported; do not adopt the reset as default. Runtime selective-load/init assertions,60-epoch histories and best-checkpoint replays passed.
