# Low-noise supervision coverage checks

128 train and 128 validation molecules, seed 0, T=100, hidden=24, layers=2, CPU. Both schedules use the same RNG policy and data. No downstream property training or test evaluation was conducted here.

| Actual training, four epochs | Old min_noise=0 | New min_noise=0.05 |
|---|---:|---:|
| t=1 sampled graphs | 11 | 11 |
| t=1 graphs with nonempty weighted supervision | 0 | 11 |
| t=1 masked atoms | 0 | 20 |
| t=1 masked physical bonds | 0 | 61 |

The first epoch sampled no t=1 training graphs in either run. One-epoch runs are retained separately to expose this rather than hide it. The four-epoch results exercise cumulative accounting; they are not independent replications of the first epoch.

`supervision.json` contains actual per-time, per-bin, per-epoch counts and cumulative training counts. Validation corruptions repeat each epoch and are not independent samples. `t1_mask_audit.json` is a separate prospective mask-only diagnostic at forced t=1, not additional training. The min_noise floor is an engineering starting point, not a tuned optimum or a guarantee of sufficient coverage. Different noise schedules change validation difficulty, so loss values must not be used to infer superiority across these runs.

The revised schedule increases t=1 base probability from about 0.02097% to 5.01974%. Clean downstream inputs still differ from noisy pretraining inputs. No property-performance conclusion follows from these checks.

Only JSON evidence is included; weights and molecular input caches remain local. Method and counter definitions: [FIXED_TOPOLOGY.md](../../motif/FIXED_TOPOLOGY.md).
