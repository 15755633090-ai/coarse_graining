# Random Region v2: H2 Base + H4 regions

The sole architecture change from v1 is the Base readout source:

- v1 `random_region_stage2_frozen`: Base Sum/Mean from H4.
- v2 `random_region_h2_stage2_frozen`: Base Sum/Mean from H2.

Both use H4 for the random region branch. Encoder source and freezing policy,
region radius 2, atoms per center 8, maximum centers 8, distance bias, attention,
five evaluation views, concatenation, and the 512 -> 128 -> 1 prediction head
are unchanged. Identical seeds initialize exactly the same parameter tensors.
The prediction head is trained from scratch; no v1 downstream weights are loaded.

The new mode is stored in model configuration and checkpoint metadata. Its
region protocol version is 2 and explicitly records Base layer 2 and coarse
source layer 4. v1's saved configuration/protocol is unchanged. Cross-mode
resume is rejected even though the parameter shapes match.

## First experiment

Finish v1 seeds 0 through 4 first. Then run only v2 seed 0. Choose subsequent
experiments using validation results; the test set is not used to tune settings.

```powershell
cd "C:\Users\12775\Desktop\GNN课题1\coarse_graining1"

conda run --no-capture-output -n polyolefin_ml python scripts/train.py `
  --dataset-root datasets/lipo `
  --task lipo `
  --mode random_region_h2_stage2_frozen `
  --encoder-init-checkpoint runs/lipo_formal_stage1_seed0_replay_v1/best.pt `
  --model-seed 0 `
  --data-seed 0 `
  --partition-seed 100000 `
  --region-radius 2 `
  --atoms-per-center 8 `
  --max-centers 8 `
  --eval-views 5 `
  --num-workers 0 `
  --output-dir runs/lipo_random_region_h2_seed0_r2_v2
```

Defaults remain batch 32, micro-batch 8, AdamW LR 0.001, weight decay 1e-5,
dropout 0.1, gradient clipping 5, BF16 training, FP32 evaluation, at most
200 epochs and patience 30. The best validation RMSE selects the checkpoint.
Append `--resume` only when resuming this v2 output directory.

## Checks

Tests verify equal v1/v2 initialization, H2 versus H4 masked Base pooling,
identical coarse representations for the same centers, frozen encoder gradients,
nonzero distance-bias gradients, v2 checkpoint reload, equal protocol presets,
and rejection of a v1 checkpoint when resuming v2.
