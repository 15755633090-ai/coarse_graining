# coarse_graining1

Dataset-independent implementation for studying global structural modules on
top of a pretrained molecular diffusion encoder, with frozen and supervised
fine-tuning modes.

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
`baseline_finetune`, and `multiscale_finetune`. The baseline preserves the
historical final-layer Sum/Mean readout; the multiscale architecture is
unchanged.

Run the two Lipo seed-0 supervised fine-tuning arms:

```powershell
python scripts/train.py --dataset-root datasets/lipo --task lipo --mode baseline_finetune --output-dir runs/lipo_seed0_ft_baseline
python scripts/train.py --dataset-root datasets/lipo --task lipo --mode multiscale_finetune --output-dir runs/lipo_seed0_ft_multiscale
```

Defaults implement the locked protocol: AdamW, batch size 32, encoder LR
`1e-5`, downstream LR `1e-3`, at most 100 epochs,
`ReduceLROnPlateau(patience=10, factor=0.3)`, and early-stopping patience 25.
Both arms start from `diffusion_encoder/encoder.pt`; no downstream checkpoint
is used for initialization.

Training writes `history.json`, `history.csv`, `best.pt`, and `last.pt`. Resume
an interrupted run with the same command plus `--resume`. `--patience`
controls early stopping on the validation selection metric.

Schema-3 checkpoints record and strictly validate the complete training
protocol (including batch size, both initial learning rates, dropout, weight
decay, scheduler, early stopping, worker count, and maximum epochs). They also
store the Git commit/dirty state, SHA256 of `encoder.pt`, and a SHA256 manifest
of the dataset source and scaffold-split files. A mismatched `--resume` is
rejected before any training step. Legacy schema-2 checkpoints remain readable
only so the existing development seed-0 run can be completed and upgraded.

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
