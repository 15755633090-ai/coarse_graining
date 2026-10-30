# Normalized size-aware readout validation

| seed | new scratch | new pretrained | pretrained minus scratch |
|---:|---:|---:|---:|
| 0 | 0.79492 | 0.76996 | -0.02496 |
| 1 | 0.81266 | 0.80293 | -0.00972 |
| 2 | 0.82268 | 0.88642 | +0.06374 |

| arm | RMSE mean ± sample SD |
|---|---:|
| scratch | 0.81009 ± 0.01405 |
| masked | 0.81977 ± 0.06002 |

Old mean-readout references: scratch0.78711 ±0.02804; masked0.81446 ±0.05114.
The readout bundle adds192 parameters to both new arms. Do not credit shared architecture gains to pretraining. All results are exploratory on reused ESOL validation; no test metrics.
