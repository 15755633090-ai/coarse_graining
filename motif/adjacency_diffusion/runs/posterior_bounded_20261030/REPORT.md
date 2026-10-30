# Posterior observation-weighting results (training max t=300)

| seed | reused scratch | original masked | posterior masked | posterior minus scratch |
|---:|---:|---:|---:|---:|
| 0 | 0.81498 | 0.87049 | 0.76875 | -0.04623 |
| 1 | 0.75889 | 0.77030 | 0.90706 | +0.14817 |
| 2 | 0.78746 | 0.80259 | 0.91151 | +0.12405 |

| arm | validation RMSE mean ± sample SD |
|---|---:|
| posterior | 0.86244 ± 0.08117 |
| scratch | 0.78711 ± 0.02804 |
| original_masked | 0.81446 ± 0.05114 |

Three exploratory seeds on reused ESOL validation. Scratch references are reused with exact clean forward/gradient equivalence checks. No test scores or isolated gate/context claims.

| seed | t | original masked AP | posterior masked AP | original true-bond type | posterior true-bond type |
|---:|---:|---:|---:|---:|---:|
| 0 | 50 | 0.56490 | 0.69438 | 0.29469 | 0.53629 |
| 0 | 200 | 0.56820 | 0.56630 | 0.27462 | 0.39278 |
| 0 | 500 | 0.52758 | 0.43990 | 0.25741 | 0.09015 |
| 0 | 1000 | 0.54210 | 0.38199 | 0.23918 | 0.12870 |
| 1 | 50 | 0.50921 | 0.72385 | 0.23576 | 0.55353 |
| 1 | 200 | 0.52255 | 0.58544 | 0.24412 | 0.32811 |
| 1 | 500 | 0.50708 | 0.47715 | 0.21857 | 0.00470 |
| 1 | 1000 | 0.49200 | 0.37721 | 0.24586 | 0.00473 |
| 2 | 50 | 0.57079 | 0.73307 | 0.32254 | 0.72321 |
| 2 | 200 | 0.56883 | 0.53113 | 0.31555 | 0.40749 |
| 2 | 500 | 0.51181 | 0.36461 | 0.29484 | 0.09305 |
| 2 | 1000 | 0.52914 | 0.35738 | 0.24863 | 0.28697 |
