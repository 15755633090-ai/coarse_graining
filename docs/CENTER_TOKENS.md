# Center-token long-range branch, version 1

Mode: `center_token_stage2_frozen`. Initialize the frozen encoder from a
validation-best Stage-1 `baseline_finetune` checkpoint for the **same task**.
Each molecule's center selection is computed once from unpadded FP32 H2/H3/H4
and saved in `center_selections.pt` and the training checkpoints. No molecule
labels, target values, property predictions, or learned branch parameters are
used during center selection.

## Fixed first-run choices

- Budget: `K = min(N, 8, max(3, ceil(N / 4)))`. For N < 3, K = N. The
  `--center-stride`, `--center-min-centers`, and `--max-centers` flags expose
  the budget for later ablations.
- Cores: one when K < 6; two otherwise. Choose the center of the largest
  connected component by minimum eccentricity, then mean graph distance, then
  atom index. With multiple components, the second core comes from the
  second-largest component. Otherwise it is the farthest candidate from the
  first core among atoms whose eccentricity is within one hop of the minimum.
- Core radius: minimum shortest-path distance to a selected core, divided by
  the largest finite such distance in the molecule. A component without a core
  gets radius 1. Near is radius <= 1/3, Mid <= 2/3, and Far > 2/3.
- Core atoms are selected Near tokens. Select one token from each nonempty
  uncovered band when the budget permits. Remaining slots follow Near:Mid:Far
  weights 3:2:1 through the current-count/weight ratio. An empty band receives
  no token, so small molecules need not instantiate all three scales.
- Within a band, first require a candidate to be at least two graph hops from
  already selected centers, if possible. If no candidate qualifies, relax the
  requirement by one hop. Rank eligible candidates by their **minimum cosine
  distance to all selected atoms** in the layer for that band: H2 for Near,
  H3 for Mid, H4 for Far. Break ties by greater minimum graph distance, then
  atom index. The core choices use topology alone.
- H2/H3/H4 centers use independent linear projections. Add a two-layer MLP of
  the normalized core radius. One global four-head self-attention block uses
  shortest-path buckets 0, 1, 2, 3, 4, 5+, and disconnected, followed by an
  FFN. Mean and Max over valid center tokens yield the long-range readout.
- Concatenate that readout with Base `[Sum(H4), Mean(H4)]` and train a new
  prediction head. Center tokens are **selected atoms**, with no region
  pooling or atom-to-region assignment.

The selection is deterministic for a fixed atom order, encoder checkpoint and
configuration. Atom-renumbering invariance is not claimed because exact ties
use atom indices. This first version covers regression tasks only.

## Checks and interpretation

The attention receives at most eight centers. A molecule with K=3 can allocate
only one center per band; a band may also be empty. Compare all center-selection
ablations using the same K and downstream network. In particular, random
centers, topology-only centers and topology-plus-feature centers separate the
contribution of feature diversity. A single dataset/seed result does not
establish that H2/H3/H4 or center selection improves generalization.

## Example: Lipo seed 0

```powershell
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'
conda run -n polyolefin_ml python scripts/train.py --dataset-root datasets/lipo --task lipo --mode center_token_stage2_frozen --encoder-init-checkpoint runs/lipo_formal_stage1_seed0_replay_v1/best.pt --model-seed 0 --data-seed 0 --partition-seed 100000 --num-workers 0 --center-stride 4 --center-min-centers 3 --center-min-separation 2 --max-centers 8 --output-dir runs/lipo_center_token_seed0_v1
```

## Example: ESOL seed 0

```powershell
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'
conda run -n polyolefin_ml python scripts/train.py --dataset-root datasets/esol --task esol --mode center_token_stage2_frozen --encoder-init-checkpoint runs/esol_seed0_stage1_base_finetune/best.pt --model-seed 0 --data-seed 0 --partition-seed 100000 --num-workers 0 --center-stride 4 --center-min-centers 3 --center-min-separation 2 --max-centers 8 --output-dir runs/esol_center_token_seed0_v1
```

Use `--resume` with the same output directory and configuration after an
interruption. The existing validation-best checkpoint rule and one-time final
test evaluation apply.
