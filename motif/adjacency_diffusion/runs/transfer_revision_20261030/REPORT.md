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
