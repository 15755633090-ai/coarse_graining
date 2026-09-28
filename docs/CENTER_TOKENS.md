# Center-token long-range branch, version 2

Mode: `center_token_stage2_frozen`. Initialize the frozen encoder from a
validation-best Stage-1 `baseline_finetune` checkpoint for the **same task**.
Each molecule's center selection is computed once from unpadded FP32 H2/H3/H4
and saved in `center_selections.pt` and the training checkpoints. No molecule
labels, target values, property predictions, or learned branch parameters are
used during center selection.

## Fixed first-run choices

- Budget: `K = min(N, 32, max(3, ceil(N / 4)))`. For N < 3, K = N. The
  `--center-stride`, `--center-min-centers`, and `--center-max-centers` flags
  expose the budget. The cap of 32 is a safety limit: none of the 5,328
  Lipo/ESOL molecules reaches it at stride 4. Attention pads only to the
  largest selected-center count **in that batch**.
- Cores are determined by graph structure, independently of K, subject to the
  total token budget. Choose the center of the largest connected component by
  minimum eccentricity, then mean graph distance, then the atom signature
  described below. In a
  disconnected graph, choose the center of the second-largest component as a
  second core when the budget permits. In a connected graph, add a second core
  only if the diameter is at least 16 hops and a candidate is at least 3 hops
  from the first core while remaining within `ceil(diameter/4)` eccentricity
  of the minimum. This prevents nearby central atoms from consuming two slots.
  Exact topology ties use an atom-order-independent signature of atom labels,
  bond labels and distances to labeled atoms before falling back to atom index.
- Core radius: minimum shortest-path distance to a selected core, divided by
  the largest finite such distance in the molecule. A component without a core
  gets radius 1. Near is radius <= 1/3, Mid <= 2/3, and Far > 2/3.
- **Topology coverage pass:** core atoms are Near tokens. Repeatedly select
  the atom with the greatest shortest-path distance to its nearest selected
  center while that distance exceeds 3 hops. A disconnected atom is treated
  as farther than any connected atom. Feature novelty breaks only exact
  distance ties. If the K budget runs out, the remaining coverage gap is
  accepted; the method cannot guarantee coverage with too few tokens.
- **Feature fill:** when all atoms are within 3 hops of a center and slots
  remain, first cover nonempty bands without a token. Other slots follow
  Near:Mid:Far weights 3:2:1. Within a band, require two-hop separation if
  possible, then rank by minimum cosine distance to selected atoms using H2
  for Near, H3 for Mid and H4 for Far. Reduce the separation requirement when
  necessary. Topology coverage can consume the whole budget, so every band is
  not guaranteed a token. Near-zero vectors receive no novelty bonus.
- H2/H3/H4 centers use independent linear projections. Add a two-layer MLP of
  the normalized core radius. One global four-head self-attention block uses
  shortest-path buckets 0, 1, 2, 3, 4, 5+, and disconnected, followed by an
  FFN. Mean and Max over valid center tokens yield the long-range readout.
- Concatenate that readout with Base `[Sum(H4), Mean(H4)]` and train a new
  prediction head. Center tokens are **selected atoms**, with no region
  pooling or atom-to-region assignment.

The selection is deterministic for a fixed graph, encoder checkpoint and
configuration. Exact chemical symmetries can still require an atom-index tie
break, so universal atom-renumbering invariance is not claimed. In a local
audit of 20 Lipo and 20 ESOL training molecules, three random permutations
each changed no prediction by more than 5e-7. This version covers regression
tasks only.

## Checks and interpretation

The attention receives at most 32 centers. A molecule with K=3 may use all
three positions for topology coverage; a band may have no token. Compare all center-selection
ablations using the same K and downstream network. In particular, random
centers, topology-only centers and topology-plus-feature centers separate the
contribution of feature diversity. A single dataset/seed result does not
establish that H2/H3/H4 or center selection improves generalization.

The `center_token_h4_stage2_frozen` control keeps the same H2/H3/H4 center
selection, Near/Mid/Far slots, three independent projections, attention,
Base readout, training protocol and parameter count. It reads **H4 for every
selected token**, so comparing it with `center_token_stage2_frozen` isolates
the H2/H3/H4 token-source choice. Give it a separate output directory. For
example, run either command below after the corresponding main run by changing
only `--mode` and `--output-dir`:

| Task | `--mode` | `--output-dir` |
|---|---|---|
| Lipo | `center_token_h4_stage2_frozen` | `runs/lipo_center_token_h4_seed0_v2` |
| ESOL | `center_token_h4_stage2_frozen` | `runs/esol_center_token_h4_seed0_v2` |

## Example: Lipo seed 0

```powershell
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'
conda run -n polyolefin_ml python scripts/train.py --dataset-root datasets/lipo --task lipo --mode center_token_stage2_frozen --encoder-init-checkpoint runs/lipo_formal_stage1_seed0_replay_v1/best.pt --model-seed 0 --data-seed 0 --partition-seed 100000 --num-workers 0 --center-stride 4 --center-min-centers 3 --center-min-separation 2 --center-coverage-radius 3 --center-max-centers 32 --output-dir runs/lipo_center_token_seed0_v2
```

## Example: ESOL seed 0

```powershell
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'
conda run -n polyolefin_ml python scripts/train.py --dataset-root datasets/esol --task esol --mode center_token_stage2_frozen --encoder-init-checkpoint runs/esol_seed0_stage1_base_finetune/best.pt --model-seed 0 --data-seed 0 --partition-seed 100000 --num-workers 0 --center-stride 4 --center-min-centers 3 --center-min-separation 2 --center-coverage-radius 3 --center-max-centers 32 --output-dir runs/esol_center_token_seed0_v2
```

Use `--resume` with the same output directory and configuration after an
interruption. The existing validation-best checkpoint rule and one-time final
test evaluation apply.
