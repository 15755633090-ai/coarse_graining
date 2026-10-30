# Direct supervised architecture result

| seed | arm | validation RMSE | training RMSE at best | active parameters | best epoch | seconds |
|---:|---|---:|---:|---:|---:|---:|
| 0 | directed | 0.945498 | 0.393825 | 80513 | 53 | 57.61 |
| 0 | boundary | 0.962090 | 0.426140 | 260996 | 37 | 94.85 |
| 1 | directed | 0.929987 | 0.449619 | 80513 | 37 | 60.22 |
| 1 | boundary | 0.935854 | 0.379510 | 260996 | 47 | 99.26 |
| 2 | directed | 0.887136 | 0.414743 | 80513 | 58 | 60.65 |
| 2 | boundary | 0.934304 | 0.451985 | 260996 | 32 | 99.89 |

{'directed': {'mean_rmse': 0.920873560586033, 'sample_sd': 0.030229635480148766, 'n': 3}, 'boundary': {'mean_rmse': 0.9440828877992581, 'sample_sd': 0.015614194481652828, 'n': 3}}

Original prior supervised reference: .787109 +/- .028044, n3.
New directed attention baseline and BRICS boundary combination differ in capacity. No independent component attribution.
No pretraining, computed descriptors, external teacher, or fingerprint features. Reused validation, no test evaluation.
Smoke runs are implementation checks, not performance evidence.
