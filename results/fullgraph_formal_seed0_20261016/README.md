# Fullgraph motif-conditioned diffusion: Lipo, seed 0

Completed experiment. Lower RMSE is better. These are single-seed paired comparisons, not evidence of a stable general improvement.

| Model | Original valid | New valid | Original test | New test | Test difference (new - original) |
|---|---:|---:|---:|---:|---:|
| HimNet | 0.724421 | 0.742251 | 0.725605 | 0.698101 | -0.027504 |
| HMG-MRL | 0.703901 | 0.695142 | 0.709210 | 0.721016 | +0.011806 |

Both paired backbones had exactly equal initial parameters. Each downstream checkpoint was selected by validation RMSE and tested after selection. HimNet new best epoch: 19 (stopped at 39); HMG-MRL new best epoch: 36 (stopped at 56). Both pretraining runs completed 50 epochs. Existing Lipo split: 3360/420/420. HMG uses the documented train-median repair of four nonfinite descriptor entries.

The method retains paper motif memberships, learns train-only shared canonical fragment type embeddings, conditions the legacy fullgraph discrete mask/replace diffusion without corrupting the type condition, then mean-pools clean fullgraph atom states into paper motif tokens/nodes. All parts are jointly fine-tuned downstream. The post-pooling projection receives downstream supervision only.

## Contents

- `comparison.json`, stage `results.json`, histories, predictions, logs, vocabulary and provenance files.
- `source/`: exact frozen experiment Python source, including the legacy diffusion package. Absolute paths in original manifests are retained as provenance.
- `artifacts/`: compressed tensors for every stage, complete molecular input cache, repaired HMG feature cache, and paper Python sources/dependency metadata.
- `related_controls/`: readable reports for the independent frozen-GINE grouping/control experiments; their full code and numerical records are in the corresponding `artifacts/gine_motif_*.zip` and `artifacts/motif_controls_*.zip`. They use different protocols and are not additional runs of this diffusion comparison.
- `FILES_SHA256.json`: byte sizes and SHA-256 for copied files and archives; this README is summarized separately from that generated manifest.

## Restore and run

Use Python 3.9+ and install `motif/requirements.txt`, `psutil`, and the corresponding paper dependencies. Original environment: PyTorch 2.8.0+cu128 and RDKit 2025.03.5. CUDA is required by the original HMG mask code.

Extract all `artifacts/*.zip` into the repository root; archives carry relative paths. If an archive is split into `.part001`, `.part002`, concatenate parts in numeric order to reconstruct the ZIP before extraction. Do not load tensor/pickle artifacts from untrusted sources.

```bash
python -m motif.check_fullgraph
python -m motif.pretrain_fullgraph --data motif/data/lipo.pt --output motif/runs/new_him_pretrain --method HimNet --epochs 50 --batch-size 4 --hidden 128 --layers 3 --seed 0
```

Full original commands are retained in `*.process.json`; replace the original workspace prefix with the clone path. For downstream runs add `--fullgraph` to `motif.train_paper` and use the corresponding fullgraph checkpoint. The `motif.launch_fullgraph` launcher can reproduce the whole sequence after restoration with a new output directory. Existing run directories must not be overwritten.

No claim that the paper motif partitions are universally ineffective follows from this comparison. The related frozen-GINE controls instead support the narrower finding that reliable grouping benefits were not established in their tested Lipo settings.
