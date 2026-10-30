# Task-side readout adaptation, with matched scratch controls

After the regularization, received-mass normalization and noise-aware weighting/range experiments failed to establish stable downstream gains, this separate exploratory experiment changes the property interface rather than adding another recovery loss. All results use the previously reused ESOL validation split; this is an adaptive study, not independent final evaluation.

## Motivation and design

The old mean readout does not explicitly receive molecule size. An encoder can still encode size indirectly (including through candidate-count message scaling), so it would be incorrect to claim that size information was entirely absent. The new interface supplies `log1p(number_of_atoms)` explicitly.

A read-only diagnostic on the same32 ESOL training molecules (no property labels used), task seed0 versus masked pretraining seed7, found mean raw graph embedding L2 norms1.44488 versus9.00931; the mean across-graph per-dimension standard deviations were0.02700 versus0.89037. This motivates testing task-side scale adaptation, but does not prove that scale caused the negative transfer or that this ratio holds across all seeds.

The property input becomes `[LayerNorm(mean(node_states)), log1p(atom_count)]`. All chemical atom/bond features, BRICS relations, boundary gates and atom states remain available to the encoder. No restoration target or supervised label is added to encoder inputs. Both scratch and pretrained arms get exactly this readout. At hidden64 the change adds192 trainable parameters relative to the old property architecture (128 LayerNorm affine parameters and64 additional first-layer weights). It is a bundled readout candidate, not an ablation that separates normalization from size.

The upstream checkpoint is the original full-range masked checkpoint from the fixed18-channel comparison. Its model weights, corruption, loss,512-molecule corpus and1200-update budget are unchanged. This experiment does not combine the posterior-weight correction with the new readout and does not claim to isolate interactions between those changes. Only encoder states are loaded; normalization and property-head parameters start fresh and are paired across both arms.

## Fixed execution

Runner `_github_sync/check_task_readout_20261030.py`; output `runs/task_readout_20261030`. Three task seeds0/1/2 map to old pretraining seeds7/1/2. Two arms per seed: scratch and original-masked pretrained. Same scaffold train/validation split, training-only target scaler, batch16, dropout0.1, Adam0.001, weight_decay0, clip5 and60 complete epochs. Initialization hashes cover encoder, property head and readout normalization; loader/dropout RNG is paired. All best checkpoints must replay their validation predictions. No test metrics, learning-rate search or seed selection.

The new checkpoint schema is `typed_masked_normalized_size_readout_v1`. Old pretraining encoders load strictly, while full new task checkpoints use the dedicated loader. Two interface tests verify that equal mean states can still supply distinct size inputs, encoder/readout gradients exist, atom permutation preserves predictions, and checkpoint roundtrip/dispatch succeeds. A two-arm CPU smoke with16 train/validation molecules and two epochs passed; it is execution acceptance, not performance evidence.

## Interpretation

The new scratch arm is the primary control for pretraining value. Compare both new arms with the previous mean-readout scratch0.78711 and masked0.81446 to assess total architecture improvement. A gain shared by both new arms is a readout/design gain; it cannot be credited to denoising. Conversely, a pretrained arm beating its new matched scratch arm on this reused validation split is still exploratory and requires independent validation before a general claim.

## Completed result

New readout scratch0.81009 ±0.01405, pretrained0.81977 ±0.06002 (three seeds). Pretrained minus same-architecture scratch is -0.02496/-0.00972/+0.06374. The final seed reverses the apparent early advantage. Neither mean beats original scratch0.78711. This readout bundle is not adopted as default and does not establish a denoising gain. All source hashes, initialization/capacity checks and checkpoint replays passed.
