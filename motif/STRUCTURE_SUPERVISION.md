# Direct supervision of the motif vector

The optional auxiliary task predicts 12 internal connectivity patterns from the
clean, t=0 motif vector `s_M` only. Its head cannot read node/edge decoder states.
The original noisy denoising objective is retained without modification.

```
python -m motif.pretrain --data motif/data/lipo.pt --output motif/runs/pretrain_structure_seed0 --structure-weight 0.1 --pair-budget 262144
```

`0.1` is a starting hyperparameter, not a validated optimum. The default weight
is zero, preserving the earlier architecture/objective and checkpoint loading.
With a positive weight, a separate clean encoding pass supplies the auxiliary
loss during training; validation reuses the existing clean recovery pass.

Labels are deterministic multi-label SMARTS matches against a graph rebuilt
solely from cached internal atom/edge tensors. No parent-molecule matching,
implicit hydrogen inference, valence completion, sanitization or outside atoms
are used. Aromatic/charge annotations already present in the cache are retained.
Unmatched motifs receive an all-zero vector. Pattern definitions are versioned
in `structure.py`, and label counts are recorded separately for train/validation.
No test motifs or downstream property labels are used by the auxiliary task.

Patterns include carbonyl, amide/carboxyl connectivity, nitrile, carbon multiple
bonds, C–O/C–N, sulfur/phosphorus–oxygen, carbon–halogen and aromatic bonds.
They do not assert functional-group identity, donor/acceptor status, electronic
effects, chemical validity, or recovery of contextual chemistry. Rare/all-zero
labels may make the task weak; positive counts must be inspected before training.
No class reweighting is estimated from validation/test data.

The loss is mean binary cross entropy over labels, then motifs:
`denoising_loss + structure_weight * clean_structure_BCE`.
`best.pt` is selected by this **combined validation objective**, not denoising
alone. History separately records validation denoising loss and structure BCE;
the checkpoint stores supervision weight and exact label definitions.
Resume requires the same protocol. Old weight-only checkpoints cannot resume a
new optimizer/objective silently. Existing paper adapters load the new config
and still consume the same t=0 `s_M`; the auxiliary head is not needed for
downstream prediction and receives no downstream-only supervision.

Checks: `python -m motif.check_structure`. A smoke run is an implementation
check, not evidence of improved representations or property prediction.

## Paired downstream entry

After formal pretraining finishes, explicitly select its validation-best weight:

```
python -m motif.launch_paper_pair --checkpoint motif/runs/pretrain_structure_seed0/best.pt --output motif/runs/paper_structure_pair_seed0
```

This command creates an immutable source snapshot and **starts training** in it,
waiting for completion. Use a new output directory for each experiment. It never
defaults to an old checkpoint or a fixed epoch. It verifies the checkpoint against
the adjacent `history.json`, rejects smoke weights and records the SHA256, actual
epoch, structure weight, and label definitions in `launch.json`/`protocol.json`.
An old checkpoint remains permitted when explicitly selected; its recorded
structure supervision weight is zero, so it is not a new-supervision experiment.

For snapshot preparation without training, add `--prepare-only`. `launch.json`
then contains the exact worker argument list and working directory for a later
manual launch. Default operation needs no `--worker` argument. `--workspace-root`
may specify the workspace containing the dataset and original paper sources.
The original HMG preprocessing completeness requirement still applies; this
entry fix does not repair missing HMG molecular features.

Routing/provenance checks: `python -m motif.check_launcher` (training subprocess
mocked; no full training or preprocessing is launched by this check).

Use `--papers HimNet` or `--papers HMG-MRL` to run only one framework while
another framework is blocked. The default remains both frameworks. Existing
HMG features may be reused via `--hmg-feature-cache PATH`; all molecules and
Motif membership mappings are still checked before training.

`repair_hmg.py` can retry only missing original HMG records with RDKit
`useRandomCoords=True, randomSeed=0`, retaining MMFF settings and the original
feature extractor. This is an explicit preprocessing deviation from the default
embedding path. It writes a separate cache, expands padding/sentinel indices
consistently, and checks full row coverage and Motif alignment. It never deletes
failed rows or overwrites the old cache. Both arms must use the same repaired
cache; record the repair manifest when reporting results.

For this Lipo cache, a separate scan found four undefined normalized descriptors
for a selenium-containing molecule (row 1561, validation). `finite_hmg.py`
replaces nonfinite descriptor entries only, using column medians computed from
unique training molecules. It writes a new cache and a manifest with affected
columns/rows, replacement values and hashes. Existing finite entries, graph
features, labels, and all split memberships are retained. This preprocessing
correction is shared by both arms and must be disclosed alongside the conformer
repair. No validation/test values are used to fit imputation statistics.

`diagnose_hmg.py` scans all cache arrays and labels for nonfinite values, then
checks initial original-model forward passes over the full validation split.
Forward hooks record the first module emitting nonfinite outputs if necessary.
`train_paper.py` now rejects nonfinite input features, predictions, auxiliary or
total loss, gradients and evaluation RMSE; runtime diagnostics record phase,
epoch and batch row IDs. A missing finite validation checkpoint cancels testing.
