# Noise-aware weighting of visible context in masked chemical-edge recovery

This is a separate correction from the regularization/aggregation screen. It does not modify that screen's frozen source, settings or selection rule.

## Concrete mismatch

The old masked encoder uses `1 - softmax(y_type)[no_bond]` to weight visible candidate messages. For a six-class clean one-hot no-bond input, its limit is `1-e/(e+5) = 0.64781257`. For a one-hot bonded class it is `1-1/(e+5) = 0.87043749`. These are the weights as positive-time noise tends to zero; the exact t=0 downstream branch instead uses absent=0, present=1. Thus even very low noise has a large topology-weight discontinuity. This is an established calculation about the implementation, not proof that it alone caused the observed negative transfer.

For the received Gaussian observation `y = sqrt(alpha_bar)*one_hot(type) + sqrt(1-alpha_bar)*epsilon`, use the local categorical likelihood:

`posterior(type | y,t) = softmax(log(training_prior) + sqrt(alpha_bar)/(1-alpha_bar) * y_type)`.

The class-independent quadratic terms cancel because all one-hot class means have equal norm. The visible message weight is `1 - posterior(no_bond)`. The prior is the same training-only Laplace-smoothed type frequency already used by the model. This is a local independent-observation model, not the full chemically constrained graph posterior. At strongly noisy observations it approaches the frequency prior; at low noise it becomes confident about visible edges/nonedges.

Fully masked target pairs receive the unchanged training presence prior, independent of the erased17 channels. The decoder still receives zero target-pair features plus the unknown flag. No target information is restored by this correction. All atom/bond attributes, BRICS prior, boundary gates, candidate-count denominator and reconstruction loss remain unchanged. In particular, this experiment does not combine posterior weighting with observation-mass normalization.

## Fixed experiment

Runner: `_github_sync/check_posterior_weighting_20261030.py`; directory `runs/posterior_weighting_20261030`. Three pretraining/task seed pairs7/0,1/1,2/2. All settings match the earlier masked experiment:512 training molecules,1200 updates, batch8, hidden64/layers3, pretraining dropout0, full random t=1..1000, mask ratio0.3, frequency initialization, Adam0.001, clip5. Independent graph/time/Gaussian/mask streams reproduce the old masked supervision counts exactly. Check equal initial trainable parameters and actual candidate-pair/true-bond counts.

Downstream uses the unchanged scaffold split, clean18-channel input, task dropout0.1, batch16, Adam0.001, no weight decay,60 full epochs and training-only target scaling. Each restored checkpoint is replayed. Confirmation recovery reports masked AP/recall/type/joint and per-attribute groups from actual pretraining heads, not fresh downstream heads.

The original scratch references are reused rather than rerun. This is justified by unchanged clean forward computation and trainable parameters: t=0 overwrites observation weights with the actual received clean support in both configurations; extra prior/schedule buffers are not trainable, and no extra RNG is consumed by posterior calculation. Per-seed task initialization matches the old head hash, and exact clean predictions/gradients under matched dropout RNG are checked before loading the pretrained encoder. This equivalence is also covered by a unit test. The reused references must remain labeled as such, not claimed as newly run scratch jobs.

There is one fixed correction and no hyperparameter search in this experiment. Its settings are frozen before execution. It is another exploratory experiment on reused ESOL validation data; it is not untouched evaluation or a paper replication. Do not claim a generalizable performance gain, isolated boundary-gate contribution or stronger context reliance without appropriate further evidence.

## Verification

Eight posterior-mode tests passed: six masking/clean-input/alignment/checkpoint/interruption tests and two low-noise/mask-prior and exact-clean-gradient checks. Low-noise visible no-bond weights are approximately0; unknown-pair weights remain the training prior and have zero gradient to erased own-pair features. Legacy softmax remains the default. The typed CLI exposes `--observation-weighting gaussian_posterior` only for `--variant masked`; `--message-normalization observation_mass` is a separate option for nonnegative typed message weights. Old checkpoint configurations receive defaults for missing optional fields.

Final results are reported only after all three runs complete. Recovery improvement and downstream improvement remain separate questions.

## Single bounded-context follow-up

After inspecting the first two full-range task seeds, a separate exploratory follow-up was fixed at training t<=300. This was an adaptive research decision, not an untouched validation confirmation. The target remains fully masked, so lower surrounding noise does not restore own-pair residual evidence. The noise schedule still has1000 steps and the same endpoint; only the sampled training range changes. No other cutoffs are searched. Use the same three pretraining/task seeds,1200 updates, source corpus, initialization, loss, supervision masks, optimizer and60-epoch downstream settings. Check identical accumulated target counts. Confirmation recovery at t500/1000 is outside the bounded training range and must be labeled accordingly. The run is `runs/posterior_bounded_20261030`; its frozen source is separate from the full-range run. The bounded result cannot by itself establish which earlier change caused any gain.

## Completed full-range result

Full-range posterior weighting: validation RMSE0.82175 ±0.02336 (three seeds), versus original masked0.81446 ±0.05114 and reused scratch0.78711 ±0.02804. Per-seed posterior minus scratch is -0.01700/+0.06369/+0.05723. Only seed0 improves. This correction fixes the low-noise probability interpretation but does not establish downstream benefit or explain negative transfer by itself. All source hashes, equal supervision counts, initialization/clean-gradient equivalence and checkpoint replays passed. The detailed masked recovery comparisons are retained in the run directory; no gain is inferred from all-pair accuracy. The separate bounded follow-up is not included in this mean.

The public typed CLI also permits an explicitly recorded task dropout change while checking all other encoder configuration fields. A one-epoch16-molecule CPU CLI acceptance loaded the older masked checkpoint with pretraining dropout0 and task dropout0.1; checkpoint replay passed. This is interface acceptance, not a performance result.

## Completed bounded result

Posterior t<=300 gives validation RMSE0.86244 ±0.08117 (three seeds). Its paired differences from reused scratch are -0.04623/+0.14817/+0.12405. The isolated seed0 improvement did not replicate; this range restriction is not adopted. All three runs used equal supervision counts and passed checkpoint/source/equivalence checks. Recovery at t500/1000 remains outside the bounded training range. Detailed recovery and task results are retained, including failures.
