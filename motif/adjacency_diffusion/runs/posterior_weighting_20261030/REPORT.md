# Posterior observation-weighting results (training max t=1000)

| seed | reused scratch | original masked | posterior masked | posterior minus scratch |
|---:|---:|---:|---:|---:|
| 0 | 0.81498 | 0.87049 | 0.79798 | -0.01700 |
| 1 | 0.75889 | 0.77030 | 0.82258 | +0.06369 |
| 2 | 0.78746 | 0.80259 | 0.84469 | +0.05723 |

| arm | validation RMSE mean ± sample SD |
|---|---:|
| posterior | 0.82175 ± 0.02336 |
| scratch | 0.78711 ± 0.02804 |
| original_masked | 0.81446 ± 0.05114 |

Three exploratory seeds on reused ESOL validation. Scratch references are reused with exact clean forward/gradient equivalence checks. No test scores or isolated gate/context claims.

| seed | t | original masked AP | posterior masked AP | original true-bond type | posterior true-bond type |
|---:|---:|---:|---:|---:|---:|
| 0 | 50 | 0.56490 | 0.56490 | 0.29469 | 0.23294 |
| 0 | 200 | 0.56820 | 0.52124 | 0.27462 | 0.22429 |
| 0 | 500 | 0.52758 | 0.52024 | 0.25741 | 0.21115 |
| 0 | 1000 | 0.54210 | 0.48785 | 0.23918 | 0.22665 |
| 1 | 50 | 0.50921 | 0.59904 | 0.23576 | 0.25968 |
| 1 | 200 | 0.52255 | 0.56708 | 0.24412 | 0.32811 |
| 1 | 500 | 0.50708 | 0.53482 | 0.21857 | 0.17391 |
| 1 | 1000 | 0.49200 | 0.52763 | 0.24586 | 0.21868 |
| 2 | 50 | 0.57079 | 0.64973 | 0.32254 | 0.28906 |
| 2 | 200 | 0.56883 | 0.55356 | 0.31555 | 0.33598 |
| 2 | 500 | 0.51181 | 0.50661 | 0.29484 | 0.22534 |
| 2 | 1000 | 0.52914 | 0.50200 | 0.24863 | 0.22563 |
