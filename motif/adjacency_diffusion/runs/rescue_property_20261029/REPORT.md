# Matched encoder transfer and readout controls (validation only)

| seed | readout | arm | best epoch | RMSE | head parameters |
|---:|---|---|---:|---:|---:|
| 0 | mean | scratch | 58 | 0.81114 | 16641 |
| 0 | mean | pretrained | 37 | 0.84668 | 16641 |
| 0 | sum | scratch | 18 | 0.96567 | 16641 |
| 0 | sum | pretrained | 20 | 0.91519 | 16641 |
| 0 | mean_sum | scratch | 37 | 0.91613 | 33153 |
| 0 | mean_sum | pretrained | 28 | 0.82156 | 33153 |
| 0 | mean_capacity | scratch | 7 | 0.93824 | 33151 |
| 0 | mean_capacity | pretrained | 12 | 0.85315 | 33151 |
| 1 | mean | scratch | 24 | 0.83701 | 16641 |
| 1 | mean | pretrained | 28 | 0.82108 | 16641 |
| 1 | mean_sum | scratch | 47 | 0.87996 | 33153 |
| 1 | mean_sum | pretrained | 32 | 0.93439 | 33153 |
| 2 | mean | scratch | 12 | 0.94292 | 16641 |
| 2 | mean | pretrained | 36 | 0.86529 | 16641 |
| 2 | mean_sum | scratch | 22 | 0.91701 | 33153 |
| 2 | mean_sum | pretrained | 30 | 0.86447 | 33153 |
