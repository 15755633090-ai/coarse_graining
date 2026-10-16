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
