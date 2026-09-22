# coarse_graining1

Clean, dataset-independent starting point for studying global structural modules on top of a frozen molecular diffusion encoder.

## Locked starting point

- `diffusion_encoder/encoder.pt` contains only the original, property-agnostic four-layer diffusion backbone.
- The encoder is always loaded frozen and one forward returns `h1`, `h2`, `h3`, and `h4`.
- No property head, property-finetuned checkpoint, coarse-graining method, hierarchy, LocalBoundary module, correction head, result, or training checkpoint is inherited.
- `datasets/` contains the original OGB molecular-property source files and published scaffold splits. Derived `processed/` caches are deliberately excluded.

The source pretraining checkpoint was:

```text
C:\Users\12775\Desktop\GNN课题1\扩散GNN\01_扩散预训练\outputs\ogb_clean\best.pt
SHA256 1358eb1baba19ccbb4463e46907e86e28085cf5b14cd39ccaef99a4fc6f6f1a9
```

## Datasets

`bace`, `bbbp`, `clintox`, `esol`, `freesolv`, `hiv`, `lipo`, `muv`, `pcba`, `sider`, `tox21`, and `toxcast` are included. The same frozen encoder and dataset interface apply to every task.

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
- `model.py` pools the frozen encoder states `h1` through `h4`, applies one MLP
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

Train the frozen-encoder first version:

```powershell
python scripts/train.py --dataset-root datasets/lipo --task lipo --output-dir runs/lipo/multiscale
```

Training writes `history.json`, `history.csv`, `best.pt`, and `last.pt`. Resume
an interrupted run with the same command plus `--resume`.

The first Lipo seed-0 run is summarized in
`results/lipo_seed0_v1/summary.json`, with its full per-epoch curve in the same
directory.
