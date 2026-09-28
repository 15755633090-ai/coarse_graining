# Fixed diffusion embedding regions: Lipo seed 0

Mode: `embedding_region_stage2_frozen`.

Use the validated Lipo fine-tuned encoder from
`runs/lipo_formal_stage1_seed0_replay_v1/best.pt`. Freeze it in eval mode;
do not repeat Stage 1 or use ESOL.

## Partition protocol

- Extract unpadded FP32 H4 for each molecule once, without corruption/noise.
- Normalize each atom vector for clustering only (epsilon 1e-12).
- Spherical k-means uses normalized centroid updates, ten uniform atom
  initializations, at most 100 iterations each, and partition seed 100000.
  Choose the restart with maximum summed cosine similarity.
- K = min(N, 8, max(2, ceil(N / 8))), exactly matching random-region token counts.
- Repair empty clusters by moving the worst represented atom from a cluster
  containing more than one atom. Zero embeddings stay zero; ties use atom order.
- Membership uses embeddings only. No graph distance, bonds, adjacency or kNN
  constrain assignment. Groups may be disconnected and are not chemical motifs.
- Cache train/valid partitions before optimization in `embedding_partitions.pt`.
  Checkpoints also carry the cache for resume/evaluation. Test partitions are
  computed on first final evaluation and included in the final checkpoint.
  Graph input hashes identify cache entries only; they are not clustering inputs.
- Encoder/config changes require a new run/cache. Atom renumbering invariance is
  not claimed for finite initialization and deterministic index tie-breaking.

## Preserved downstream model and comparison limits

Keep original, unnormalized H4 Sum/Mean region pooling, region MLP, four-head
attention, distance bias, FFN, Base H4 Sum/Mean, concatenation and prediction head.
Select the atom nearest each spherical centroid (among that cluster's members)
as its representative. Original graph distances between these representatives
feed the existing attention bias **after** membership has been decided.

The random control permits overlap and incomplete coverage; k-means covers
every atom exactly once. Consequently this compares partition schemes including
their coverage behavior, not cosine information alone. Random evaluation uses
five distinct views; the fixed partition has one view (repeating it in eval mode
would yield identical predictions). `eval_views` and `region_radius` have no
effect on this mode. No extra trainable parameters are introduced.

Training hyperparameters, data split, scaler and validation checkpoint selection
follow the existing frozen Stage-2 protocol. A seed-0 result is preliminary.

## Run

```powershell
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'
conda run -n polyolefin_ml python scripts/train.py --dataset-root datasets/lipo --task lipo --mode embedding_region_stage2_frozen --encoder-init-checkpoint runs/lipo_formal_stage1_seed0_replay_v1/best.pt --model-seed 0 --data-seed 0 --partition-seed 100000 --num-workers 0 --atoms-per-center 8 --max-centers 8 --eval-views 1 --output-dir runs/lipo_embedding_region_seed0_v1
```

The shell-local MKL setting avoids the installed environment's conflicting
OpenMP runtimes; it does not change any installed libraries.

Append `--resume` only for that same configuration/output directory.
