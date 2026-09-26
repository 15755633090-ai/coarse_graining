# Random region interaction, version 1

This experiment keeps the H4 Sum/Mean Base representation and concatenates
an additional region representation. It does not use the old multiscale
partitions, coarse edges, or residual prediction head.

## Fixed first-run configuration

- Mode: `random_region_stage2_frozen`, Lipo regression, model/data seed 0.
- Source: `runs/lipo_formal_stage1_seed0_replay_v1/best.pt`, encoder only.
- Encoder remains frozen and in evaluation mode. The concatenated prediction
  head is trained from scratch, as in the Stage-2 Base comparison.
- Radius 2; K = min(N, 8, max(2, ceil(N / 8))).
- Uniform sampling without replacement; overlapping regions are allowed.
- Region H4 Sum/Mean -> two-layer MLP -> 128-dimensional tokens.
- One pre-norm attention block, four heads, residual connections and a
  two-layer feed-forward network. Dropout 0.1 follows the Stage-2 preset.
- Per-head learned distance bias; buckets 0 through 8, 9 through 12,
  13 or more, and disconnected. Bias starts at zero and is trainable.
- Token Sum/Mean -> concatenate with Base Sum/Mean -> prediction head.
- Training uses one view per sample per epoch (each sample is visited once).
  Seeds depend on sample ID and epoch, independently of model/loader RNG.
- Validation and test average five predictions using sample-ID/view seeds.
  The encoder is evaluated once per batch and reused for all views.
- AdamW, LR 0.001, weight decay 1e-5, batch 32, micro-batch 8, gradient
  clipping 5, BF16 training and FP32 evaluation. Maximum 200 epochs,
  patience 30; validation RMSE selects the checkpoint. Test runs once at end.
- A single atom yields K=1; padded atoms are excluded from sampling/pooling.
  Distances follow chemical bonds, with a separate disconnected bucket.

The first version exposes only regression modes because it averages raw
regression predictions. Classification probability averaging is not implemented.
The retained Base path includes every atom but is not a lossless graph encoding.
Sampling by atom indices is invariant in distribution to atom renumbering;
finite fixed views are not guaranteed to give identical predictions after
renumbering. The dataset atom ordering is fixed for this experiment.

## Train-split geometry

The audit uses all 3,360 training molecules and five fixed views without using
labels. Median atom count is 27, and median K is 4.

| Statistic | r=2 | r=4 |
|---|---:|---:|
| Mean region / molecule atom count | 23.33% | 45.27% |
| Mean pairwise Jaccard | 13.95% | 32.48% |
| Identical region pair fraction | 0.35% | 1.90% |
| Regions covering at least 70% of molecule | 0.20% | 8.84% |

These statistics motivate choosing r=2 before training. They do not establish
which radius produces better prediction accuracy.

## Commands

```powershell
conda run -n polyolefin_ml python scripts/analyze_random_regions.py
conda run -n polyolefin_ml python -m unittest discover -s tests -p test_random_regions.py
conda run -n polyolefin_ml python scripts/train.py --dataset-root datasets/lipo --task lipo --mode random_region_stage2_frozen --encoder-init-checkpoint runs/lipo_formal_stage1_seed0_replay_v1/best.pt --model-seed 0 --data-seed 0 --num-workers 0 --region-radius 2 --atoms-per-center 8 --max-centers 8 --eval-views 5 --output-dir runs/lipo_random_region_seed0_r2_v1
```

Append `--resume` to resume that exact run. Region settings are saved in the
checkpoint protocol and checked on resume. Use a new output directory for
any changed experiment.

The matched Stage-2 Base has validation RMSE 0.67518 and test RMSE 0.69381.
The stronger Stage-1 Base has validation RMSE 0.67109 and test RMSE 0.68153.
A seed-0 improvement is preliminary evidence, not a stability claim.
