# coarse_graining1

Dataset-independent implementation for studying global structural modules on
top of a pretrained molecular diffusion encoder, with frozen and supervised
fine-tuning modes.

## Random region experiment

The simplified H4 random-region branch is available as
`random_region_stage2_frozen`. It retains the Base Sum/Mean representation,
adds radius-2 region tokens with one distance-aware attention block, and
concatenates both readouts. Validation/test average five fixed views.
Configuration, geometry statistics, and the seed-0 command are documented in
[`docs/RANDOM_REGIONS.md`](docs/RANDOM_REGIONS.md).

## Locked starting point

- `diffusion_encoder/encoder.pt` contains only the original, property-agnostic four-layer diffusion backbone.
- The untouched pretrained encoder returns `h1`, `h2`, `h3`, and `h4`; the
  experiment mode determines whether property gradients update it.
- No property head, property-finetuned checkpoint, coarse-graining method, hierarchy, LocalBoundary module, correction head, result, or training checkpoint is inherited.
- `datasets/` contains the original OGB molecular-property source files and published scaffold splits. Derived `processed/` caches are deliberately excluded.

The source pretraining checkpoint was:

```text
C:\Users\12775\Desktop\GNN课题1\扩散GNN\01_扩散预训练\outputs\ogb_clean\best.pt
SHA256 1358eb1baba19ccbb4463e46907e86e28085cf5b14cd39ccaef99a4fc6f6f1a9
```

## Datasets

`bace`, `bbbp`, `clintox`, `esol`, `freesolv`, `hiv`, `lipo`, `muv`, `pcba`, `sider`, `tox21`, and `toxcast` are included. The same pretrained encoder and dataset interface apply to every task.

Run the project in the existing `polyolefin_ml` environment. Dependencies are
listed in `requirements.txt`; partition canonicalization uses `python-igraph`.

## Minimal usage

```python
from torch.utils.data import DataLoader

from diffusion_encoder.data import (
    OGBMoleculePropertyDataset,
    collate_property_records,
)
from diffusion_encoder.encoder import load_frozen_encoder

dataset = OGBMoleculePropertyDataset("datasets/lipo", split="train")
loader = DataLoader(dataset, batch_size=32, collate_fn=collate_property_records)
encoder = load_frozen_encoder("diffusion_encoder/encoder.pt", device="cuda")

batch = next(iter(loader)).to("cuda")
h1, h2, h3, h4 = encoder(
    batch.graph.node_features,
    batch.graph.bonds,
    batch.graph.node_mask,
)
```

## Multiscale tokenization

The first-version method is implemented in `multiscale_tokenizer/`:

- `partition.py` samples the four-scale partition `P(G; omega)`, including the
  iterative level-4 outer rounds and local residual tokens;
- `model.py` pools encoder states `h1` through `h4`, applies one MLP
  per scale, and predicts from `z_G = z_Q + z_2 + z_3 + z_4`;
- `training.py` keeps partition seeds separate from model and data-loader
  seeds. Training redraws partitions by epoch; validation and test use the
  fixed epoch-0 partition.

The exact algorithm contract is in `docs/ALGORITHM.md`.

```python
from multiscale_tokenizer.model import MultiscaleMolecularModel

model = MultiscaleMolecularModel(device="cuda")
output = model(
    batch.graph.node_features,
    batch.graph.bonds,
    batch.graph.node_mask,
    partition_seeds=[101, 102],
)
```

Run the partition/statistical audit before training:

```powershell
python scripts/analyze_tokenization.py --dataset-root datasets/lipo --split train --limit 100
```

The training entry point supports `baseline_frozen`, `multiscale_frozen`,
`baseline_finetune`, `multiscale_finetune`, `baseline_stage2_frozen`, and
`multiscale_stage2_frozen`. The baseline preserves the historical final-layer
Sum/Mean readout; the multiscale architecture is unchanged.

The training protocol is locked to the earlier formal benchmark that produced
the GROVER / D-MPNN / GINE / PNA / AttentiveFP comparison numbers. Both stages
use the published scaffold split, at most 200 epochs, early-stopping patience
30, AdamW with weight decay `1e-5`, gradient clipping 5.0, BF16 autocast on
CUDA, micro-batch gradient accumulation of 8, a constant learning rate, a
target scaler fitted on the training split only, and validation-RMSE model
selection. The best-validation checkpoint is evaluated on the test split once.

Stage 1 (`baseline_finetune`, `multiscale_finetune`) uses batch 64, encoder LR
`1e-4`, downstream LR `5e-4` and dropout 0.3. Stage 2 and frozen modes use
batch 32, a frozen encoder, downstream LR `1e-3` and dropout 0.1. The entry
point resolves those values from the mode, so the paired Stage-2 arms cannot
drift apart.

Training execution is ported from the earlier `downstream_benchmark.py` loop:
per-batch micro-batch accumulation (micro 8, with padding re-tightened per
micro-batch), BF16 autocast, grad clipping, a full validation pass under
`torch.no_grad()`, validation-RMSE selection, a host-side copy of the selected
state, and early stopping. The old loop recorded no per-batch level/token
diagnostics during training, so the formal path skips them as well; use
`--train-diagnostics` only for method-development runs that need those
per-micro statistics.

Run the Stage-1 Base seed-0 reproduction check:

```powershell
python scripts/train.py `
  --dataset-root datasets/lipo `
  --task lipo `
  --mode baseline_finetune `
  --model-seed 0 `
  --data-seed 0 `
  --num-workers 0 `
  --output-dir runs/lipo_formal_stage1_seed0
```

After the Stage-1 encoder has recovered the earlier range, run the paired
frozen comparison from the same Stage-1 best checkpoint:

```powershell
python scripts/train.py `
  --dataset-root datasets/lipo `
  --task lipo `
  --mode baseline_stage2_frozen `
  --encoder-init-checkpoint runs/lipo_formal_stage1_seed0/best.pt `
  --model-seed 0 `
  --data-seed 0 `
  --num-workers 0 `
  --output-dir runs/lipo_formal_stage2_seed0_baseline

python scripts/train.py `
  --dataset-root datasets/lipo `
  --task lipo `
  --mode multiscale_stage2_frozen `
  --encoder-init-checkpoint runs/lipo_formal_stage1_seed0/best.pt `
  --model-seed 0 `
  --data-seed 0 `
  --num-workers 0 `
  --output-dir runs/lipo_formal_stage2_seed0_multiscale
```

The five formal seeds are `0` through `4`; pass the same value to
`--model-seed` and `--data-seed`, and keep `--num-workers` identical across
the paired Stage-2 arms.

These commands load only the Stage-1 checkpoint's `encoder.*` tensors. They
freeze that encoder and initialize a new downstream readout at epoch 1; they
do not resume Stage 1 or reuse its prediction head. The source checkpoint hash
and its actual saved model-state epoch are recorded in Stage-2 provenance.
New checkpoints explicitly record `model_state_epoch`. Legacy checkpoints are
audited against their producer commit and old local-improvement behavior; an
ambiguous checkpoint or one whose saved state is not the reported validation
best is rejected as a Stage-2 source.

Base execution revision `legacy_base_replay_v1` also reproduces the original
property-head initialization RNG stream: temporary CPU Linear initializations
consume the draws formerly used by discarded pretraining decoders. No decoder
is attached to the model or loaded into downstream training. Training uses
BF16; validation and test use FP32. Deterministic algorithms are enabled with
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, and evaluation loaders use independent
generators seeded with `data_seed + 10000` and `data_seed + 20000`.

Use a new output directory after this correction (for example,
`runs/lipo_formal_stage1_seed0_replay_v1`). Earlier checkpoints remain readable
but cannot resume into the revised execution protocol. Earlier `legacyexec`
results do not establish reproduction of the original Base benchmark.

On the machine containing the original benchmark source and checkpoint, run:

```powershell
python scripts/audit_base_execution.py --replay-epochs 2
```

This compares encoder/head initialization, RNG state, the first real shuffled
Lipo batch, scaler, CUDA training predictions, clipped gradients, AdamW update,
and FP32 evaluation output. It then compares two validation epochs against
the original saved history without evaluating the test split. The report is
written to `results/base_execution_audit.json`. Full-run metric reproduction
still requires a fresh Stage-1 run.

Training writes `history.json`, `history.csv`, `best.pt`, and `last.pt`. Resume
an interrupted run with the same command plus `--resume`. `--patience`
controls early stopping on the validation selection metric.

Schema-4 checkpoints record and strictly validate the complete training
protocol (including batch and micro-batch size, both learning rates, dropout,
weight decay, gradient clipping, AMP dtype, scheduler, early-stopping metric
and patience, worker count, and maximum epochs). They also store the Git
commit/dirty state, SHA256 of `encoder.pt`, and a SHA256 manifest of the
dataset source and scaffold-split files. A mismatched `--resume` is rejected
before any training step. Legacy schema-2 and schema-3 checkpoints remain
readable, but their recorded protocol no longer matches the formal benchmark,
so they cannot be resumed under the locked defaults.

Audit the code protocol against the earlier formal `selected_config.json`
before starting a run:

```powershell
python scripts/audit_protocol.py
```

DataLoader workers are persistent across epochs. A shared epoch value redraws
training partitions without restarting Windows worker processes, while
validation/test remain fixed at epoch 0. The loader explicitly preserves the
former per-epoch base-seed consumption, so this performance optimization does
not shift the sampler or model RNG trajectory when resuming a checkpoint.

Evaluate the saved best checkpoint without retraining:

```powershell
python scripts/evaluate.py `
  --checkpoint runs/lipo_seed0_stage1/best.pt `
  --dataset-root datasets/lipo `
  --task lipo `
  --split test
```

The first Lipo seed-0 run is summarized in
`results/lipo_seed0_v1/summary.json`, with its full per-epoch curve in the same
directory.

Stochastic partitions are computed in DataLoader workers and the GPU pools all
tokens for the batch with batched `index_add` operations. The measured Lipo
pipeline improvement is recorded in
`results/performance/lipo_training_pipeline.json`.
