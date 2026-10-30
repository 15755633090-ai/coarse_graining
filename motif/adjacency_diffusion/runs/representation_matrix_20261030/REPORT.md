# Objective x corpus representation study

| corpus | objective | validation RMSE mean ± sample SD |
|---|---|---:|
| small | edge | 0.81446 ± 0.05114 |
| small | representation | 0.83449 ± 0.03540 |
| expanded | edge | 0.84288 ± 0.01684 |
| expanded | representation | 0.82339 ± 0.01466 |

Original scratch reference: 0.78711 ± 0.02804.

| pretrain seed | objective Δsmall | objective Δexpanded | data Δedge | data Δrepresentation |
|---:|---:|---:|---:|---:|
| 7 | +0.00477 | -0.02634 | -0.01053 | -0.04164 |
| 1 | +0.04121 | -0.01969 | +0.05599 | -0.00490 |
| 2 | +0.01412 | -0.01245 | +0.03980 | +0.01324 |

Negative differences are better. Paired graph draws/updates, not strict compute parity. Same-source ZINC expansion is not domain alignment. All validation splits have been reused; no test evaluation.

| source | clean effective rank | clean dimension std | unique molecules seen | encoder graph passes |
|---|---:|---:|---:|---:|
| small_edge_7 | 3.5740 | 0.5336 | original | 9600, original |
| small_representation_7 | 1.0001 | 0.7080 | 512 | 19200 |
| small_edge_1 | 2.9480 | 0.6138 | original | 9600, original |
| small_representation_1 | 1.0002 | 0.7274 | 512 | 19200 |
| small_edge_2 | 3.7400 | 0.4980 | original | 9600, original |
| small_representation_2 | 1.0001 | 0.7029 | 512 | 19200 |
| expanded_edge_7 | 2.9234 | 0.6236 | 3710 | 9600 |
| expanded_representation_7 | 1.0001 | 0.6928 | 3710 | 19200 |
| expanded_edge_1 | 3.1569 | 0.5573 | 3711 | 9600 |
| expanded_representation_1 | 1.0004 | 0.6820 | 3711 | 19200 |
| expanded_edge_2 | 2.6142 | 0.5853 | 3668 | 9600 |
| expanded_representation_2 | 1.0001 | 0.7555 | 3668 | 19200 |

Low rank is not constant collapse and does not by itself prove task uselessness. The small-batch graph adaptation is not the vision LeJEPA recipe. Untrained chemical recovery heads are not evaluated as representation recovery scores.
