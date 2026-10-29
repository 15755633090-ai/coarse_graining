# Transfer diagnosis: completed controlled comparisons

| arm | original early stop | all 60 epochs | low encoder LR, original early stop |
|---|---:|---:|---:|
| scratch | 0.81658 | 0.79666 | 0.95707 |
| full | 0.86259 | 0.81827 | not run |
| bounded | 0.85614 | 0.84200 | 1.03764 |

All metrics are validation RMSE. Original files remain unchanged.
Full and bounded use identical paired pretraining settings except their noise horizons.
Both pretraining arms remain worse than scratch in all three seeds after all 60 epochs.
Bounded is worse than full in all three seeds at the common 60-epoch cap.
Lower-LR experiments retain early stopping and do not identify every possible optimizer policy.
Frozen probes demonstrate usable information, not a downstream gain.
No test set was evaluated; causal claims are limited to the tested interventions.
