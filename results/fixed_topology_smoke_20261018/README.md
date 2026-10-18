# Fixed-topology implementation smoke evidence

These JSON records were copied from the local 2026-10-18 checks. They validate execution, not predictive benefit.

- Pretraining: 32 train and 32 validation molecules, one epoch; no test optimization/selection.
- Each paper: original/random/pretrained, seed 0, two molecules per split, one epoch. Validation selects the checkpoint before final test evaluation.
- Both completed.json files record identical original-backbone initialization across the three arms.
- Adapter checks use actual paper model classes with synthetic inputs and check nonzero encoder gradients.

Source code and methodological limits: [motif/FIXED_TOPOLOGY.md](../../motif/FIXED_TOPOLOGY.md).

Weights, input datasets, feature caches, and third-party repositories are not included in this evidence directory. Local paths and source hashes in manifests describe the original execution environment. Do not interpret smoke RMSE values as formal comparisons.
