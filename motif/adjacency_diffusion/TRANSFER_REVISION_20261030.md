# Bounded transfer revision: regularization and observed-mass aggregation

## Evidence and hypotheses

The preceding fixed 18-input-channel experiment completed 6 upstream and 9 downstream runs. Three-seed ESOL validation RMSE was scratch 0.78711 ± 0.02804, unmasked pretraining 0.79606 ± 0.00595, and masked pretraining 0.81446 ± 0.05114. No test labels were evaluated. This revision does not treat reduced restoration accuracy as a desired outcome.

Masked seeds 0 and 2 peaked at epoch12; seed1 peaked at epoch28. Training losses continued to decrease while final validation errors exceeded the best values. This supports late-stage downstream overfitting, but does not establish that upstream overfitting caused negative transfer. The scratch and unmasked arms also had worse final than best validation scores. Train batch loss and validation RMSE have different scales and are not directly compared.

The prior reducer divides weighted messages by all candidate neighbors. Corrupted typed inputs give many candidates positive weights; the clean t=0 path has zero weight on absent bonds. The denominator stays the candidate count in both paths. This creates a potential aggregation-scale mismatch. The new optional `observation_mass` reducer divides by the sum of received observation weights within the same allowed relation, with an epsilon guard. It adds no parameters, uses no hidden clean target support, and is a hypothesis to test rather than an established explanation. Conditional gates remain message multipliers and are not included in the denominator. This mode is used for nonnegative typed probability weights, not signed binary Gaussian weights.

The default `candidate` path is retained for existing experiments and checkpoints. The clean atom features, all bond attributes, BRICS conditions, internal/context states and boundary gates are unchanged. Pretraining still completely masks target-pair observations; it does not restore an own-pair copying path.

## Frozen bounded search

The runner is `_github_sync/search_transfer_revision_20261030.py`. Source files and hashes are copied into the run's `frozen_source` before execution. No architecture, corpus or budget changes are made during the screen.

| candidate | task learning rate | weight decay | dropout | message reducer |
|---|---:|---:|---:|---|
| slow_regularized | 0.0003 | 0.0001 | 0.2 | candidate count |
| strong_regularized | 0.001 | 0.001 | 0.3 | candidate count |
| observation_mass | 0.001 | 0 | 0.1 | received weight sum |

Every candidate is run with both scratch and masked initialization on task seed0, for exactly60 epochs, batch16, clip5, the original ESOL scaffold train/validation split, training-only label scaling and the same mean readout. A candidate changes a predefined regularization bundle, so it does not identify the independent effect of each hyperparameter. No learning-rate schedule or early stopping is introduced. Encoder and head initialization hashes are paired; loader order and task RNG are paired. All best checkpoints are replayed.

The two regularization candidates reuse the original masked 1200-step checkpoints. For observed-mass aggregation, masked pretraining is rerun with the same512-molecule corpus, 1200 updates, batch8, hidden64/layers3, dropout0, Adam0.001, frequency initialization, random t=1..1000, and ratio0.3. Graph/time/Gaussian/mask RNG streams are independently paired to the prior masked experiment. The normalization is changed consistently in pretraining and downstream. This tests the full aggregation design, not an isolated posthoc normalization patch. For the scratch arm, no pretrained encoder weights are loaded.

Selection is fixed in advance: choose the lowest masked validation RMSE on screening seed0, then run that candidate's scratch/masked pair on task seeds1 and2 (pretraining seeds1 and2). No second selection is allowed after these replications. The selected three-seed mean includes the selected seed and is optimistic; report seed1/2 separately. These are seed replications on a reused validation split, not an independent dataset or untouched test evaluation. All failed candidates must be retained in the report. Improvement over the old scratch result cannot by itself establish a pretraining gain: the new matched scratch result is the primary comparison.

## Verification and limits

The original26 interface/masked/typed/refined tests passed after adding the optional reducer. Eight new-mode tests passed: two cover exact weighted means, zero support, observation-weight gradients, atom permutation and checkpoint replay; six rerun the masking, clean-input, alignment and interruption-recovery checks under observation-mass normalization. A six-run CPU smoke exercise passed; it is execution validation only. No context intervention or isolated gate ablation is part of this search. Final status/results must come from the run artifacts; no performance claim is implied by these checks.

## Completed result

# Bounded transfer revision results

All validation RMSE values below use60 full epochs; lower is better.

| phase | candidate | seed | scratch | masked | masked minus scratch |
|---|---|---:|---:|---:|---:|
| replication | strong_regularized | 1 | 0.97650 | 0.83535 | -0.14115 |
| replication | strong_regularized | 2 | 0.94906 | 0.81740 | -0.13165 |
| screen | observation_mass | 0 | 1.19030 | 1.12863 | -0.06167 |
| screen | slow_regularized | 0 | 0.88090 | 0.95649 | +0.07559 |
| screen | strong_regularized | 0 | 0.96720 | 0.88435 | -0.08285 |

Selected using masked seed0: strong_regularized.

Selected aggregate (includes the seed used for selection):
{"scratch": {"n": 3, "mean_rmse": 0.9642539759335742, "sd": 0.013959760471407479}, "masked": {"n": 3, "mean_rmse": 0.8457018456712725, "sd": 0.034649873030412524}}

Replication seeds1,2 only:
{"scratch": {"n": 2, "mean_rmse": 0.9627804898498762, "sample_sd": 0.019409349741942177}, "masked": {"n": 2, "mean_rmse": 0.8263794518596879, "sample_sd": 0.012692438157723297}}

All candidates are disclosed. This is a small exploratory search on a reused validation split. No test evaluation, no isolated gate or context-causality claim.

The selected regularized masked mean0.84570 remains worse than the earlier original masked0.81446 and original scratch0.78711. Its advantage over the heavily regularized scratch0.96425 does not improve the best available prediction result. The slow/regularized and observed-mass candidates also failed the seed0 screen against the original configurations. Do not change the default reducer or adopt these regularization bundles on this evidence. A separate Gaussian-posterior weighting experiment addresses a more specific low-noise topology discontinuity; it is not merged into this screen.
