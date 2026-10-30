# Direct supervised architecture result

| seed | arm | validation RMSE | training RMSE at best | active parameters | best epoch | seconds |
|---:|---|---:|---:|---:|---:|---:|
| 0 | directed | 2.047446 | 2.304210 | 5425 | 1 | 0.59 |
| 0 | boundary | 2.053233 | 2.307225 | 9554 | 1 | 0.11 |

{'directed': {'mean_rmse': 2.047446251281334, 'sample_sd': None, 'n': 1}, 'boundary': {'mean_rmse': 2.0532329327737773, 'sample_sd': None, 'n': 1}}

Original prior supervised reference: .787109 +/- .028044, n3.
New directed attention baseline and BRICS boundary combination differ in capacity. No independent component attribution.
No pretraining, computed descriptors, external teacher, or fingerprint features. Reused validation, no test evaluation.
Smoke runs are implementation checks, not performance evidence.
